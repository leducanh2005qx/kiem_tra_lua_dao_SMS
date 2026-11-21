"""Train, select, evaluate and serialize the Phase 1 SMS models."""
import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
import sklearn

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

from src.config import LABEL_NAMES, MODEL_DIR, PROCESSED_DIR, RAW_DIR, RESULTS_DIR, SEED
from src.data_utils import load_sms_csv, remove_exact_test_overlap, sha256_file
from src.modeling import (
    NB_BASELINE,
    NB_CANDIDATES,
    SVM_BASELINE,
    SVM_CANDIDATES,
    get_model_spec,
    make_pipeline,
)
from src.preprocess import redact_message
from src.robustness import robustness_frame


CANDIDATES = [NB_BASELINE, *NB_CANDIDATES, SVM_BASELINE, *SVM_CANDIDATES]


def metric_row(name, subset, true, predicted, family=None, method=None, fit_seconds=None):
    """Return required aggregate, positive-class and false-negative metrics."""
    true = np.asarray(true)
    predicted = np.asarray(predicted)
    precision_1 = precision_score(true, predicted, pos_label=1, zero_division=0)
    recall_1 = recall_score(true, predicted, pos_label=1, zero_division=0)
    f1_1 = f1_score(true, predicted, pos_label=1, zero_division=0)
    return {
        "model": name,
        "family": family or name,
        "method": method or name,
        "split": subset,
        "samples": int(len(true)),
        "accuracy": float(accuracy_score(true, predicted)),
        "precision_macro": float(precision_score(true, predicted, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(true, predicted, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(true, predicted, average="macro", zero_division=0)),
        "precision_label_1": float(precision_1),
        "recall_label_1": float(recall_1),
        "f1_label_1": float(f1_1),
        "false_negative_label_1": int(((true == 1) & (predicted == 0)).sum()),
        # Keep compatibility with the existing Streamlit page.
        "precision_scam": float(precision_1),
        "recall_scam": float(recall_1),
        "f1_scam": float(f1_1),
        "fit_seconds": round(float(fit_seconds), 4) if fit_seconds is not None else None,
    }


def _model_label(name):
    return {
        "nb_word_1": "NB baseline",
        "nb_optimized": "NB tối ưu",
        "svm": "Linear SVM",
    }.get(name, name)


def _select_best(rows, candidates):
    """Select Macro-F1 first, then positive recall, then precision, deterministically."""
    rank = {name: index for index, name in enumerate(candidates)}
    candidate_rows = [row for row in rows if row["method"] in candidates]
    if not candidate_rows:
        raise ValueError("Không có kết quả ứng viên để chọn mô hình.")
    return max(
        candidate_rows,
        key=lambda row: (
            row["macro_f1"],
            row["recall_label_1"],
            row["precision_label_1"],
            -rank[row["method"]],
        ),
    )["method"]


def _fit_and_score(method, x_train, y_train, x_eval, y_eval, subset):
    started = time.perf_counter()
    model = make_pipeline(method).fit(x_train, y_train)
    elapsed = time.perf_counter() - started
    prediction = model.predict(x_eval)
    spec = get_model_spec(method)
    row = metric_row(method, subset, y_eval, prediction, spec["family"], method, elapsed)
    row.update(spec)
    return model, prediction, row


def _save_feature_explanations(models, model_dir, results_dir):
    output_dir = results_dir / "feature_explanations"
    output_dir.mkdir(parents=True, exist_ok=True)
    for artifact_name, model in models.items():
        vectorizer = model.named_steps["tfidf"]
        classifier = model.named_steps["clf"]
        names = vectorizer.get_feature_names_out()
        if hasattr(classifier, "coef_"):
            weights = np.asarray(classifier.coef_[0])
            method = "linear margin coefficient"
        else:
            weights = np.asarray(classifier.feature_log_prob_[1] - classifier.feature_log_prob_[0])
            method = "NB log probability difference"
        order = np.argsort(weights)
        indices = list(order[:10]) + list(order[-10:][::-1])
        rows = [
            {
                "model": artifact_name,
                "feature": str(names[index]),
                "weight": round(float(weights[index]), 8),
                "toward_label": 1 if weights[index] > 0 else 0,
                "explanation_method": method,
            }
            for index in indices
        ]
        pd.DataFrame(rows).to_csv(output_dir / f"{artifact_name}.csv", index=False, encoding="utf-8-sig")


def _save_figures(train, metrics, predictions, y_true, results_dir):
    figures_dir = results_dir / "figures"
    confusion_dir = results_dir / "confusion_matrix"
    figures_dir.mkdir(parents=True, exist_ok=True)
    confusion_dir.mkdir(parents=True, exist_ok=True)

    counts = train["label"].value_counts().reindex([0, 1], fill_value=0)
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar([LABEL_NAMES[k] for k in counts.index], counts.values)
    ax.set_title("Phân bố nhãn trong train sau làm sạch")
    ax.set_ylabel("Số SMS")
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), str(int(bar.get_height())), ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(figures_dir / "class_distribution.png", dpi=170)
    plt.close(fig)

    plot_metrics = metrics.copy()
    plot_metrics["display"] = plot_metrics["model"].map(_model_label)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(plot_metrics["display"], plot_metrics["macro_f1"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Macro-F1 (test)")
    ax.set_title("So sánh mô hình trên tập test giữ lại")
    for bar, value in zip(bars, plot_metrics["macro_f1"]):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.01, f"{value:.3f}", ha="center")
    fig.tight_layout()
    fig.savefig(figures_dir / "model_comparison.png", dpi=170)
    plt.close(fig)

    for model_name, y_pred in predictions.items():
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        image = ax.imshow(cm, cmap="Blues")
        ax.set_xticks([0, 1], ["Hợp lệ (0)", "Rác/nghi vấn (1)"])
        ax.set_yticks([0, 1], ["Hợp lệ (0)", "Rác/nghi vấn (1)"])
        ax.set_xlabel("Dự đoán")
        ax.set_ylabel("Thực tế")
        ax.set_title(f"Ma trận nhầm lẫn – {_model_label(model_name)}")
        for row in range(2):
            for col in range(2):
                ax.text(col, row, str(cm[row, col]), ha="center", va="center")
        fig.colorbar(image, ax=ax)
        fig.tight_layout()
        fig.savefig(confusion_dir / f"{model_name}.png", dpi=170)
        # Keep the prior app/report location available for compatibility.
        fig.savefig(figures_dir / f"confusion_{model_name}.png", dpi=170)
        plt.close(fig)


def _save_robustness(models, results_dir):
    cases = robustness_frame()
    output = cases[["case_id", "reference_label", "transformation_type"]].copy()
    output["message_original"] = cases["original"].map(redact_message)
    output["message_transformed"] = cases["transformed"].map(redact_message)
    for model_name, model in models.items():
        original_pred = model.predict(cases["original"])
        transformed_pred = model.predict(cases["transformed"])
        output[f"{model_name}_original_prediction"] = original_pred
        output[f"{model_name}_transformed_prediction"] = transformed_pred
        output[f"{model_name}_transformed_correct"] = transformed_pred == cases["reference_label"].to_numpy()
        output[f"{model_name}_prediction_changed"] = original_pred != transformed_pred
    output.to_csv(results_dir / "robustness_test_results.csv", index=False, encoding="utf-8-sig")


def run(train_file, test_file, model_dir, results_dir, processed_dir):
    model_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    train, train_quality = load_sms_csv(train_file)
    test, test_quality = load_sms_csv(test_file)
    train, overlap_unique, overlap_rows = remove_exact_test_overlap(train, test)
    if len(train) < 20 or len(test) < 6:
        raise ValueError("Cần >=20 mẫu train và >=6 mẫu test sau xử lý.")
    if train["label"].nunique() != 2 or test["label"].nunique() != 2:
        raise ValueError("Train và test đều phải có hai nhãn 0, 1.")
    if train["label"].value_counts().min() < 4:
        raise ValueError("Mỗi lớp trong train cần ít nhất 4 mẫu.")

    x_fit, x_val, y_fit, y_val = train_test_split(
        train["message"], train["label"], test_size=0.2, random_state=SEED, stratify=train["label"]
    )
    print("Train-fit:", len(x_fit), "Validation:", len(x_val), "Test:", len(test))

    validation_rows = []
    for method in CANDIDATES:
        _, _, row = _fit_and_score(method, x_fit, y_fit, x_val, y_val, "validation")
        validation_rows.append(row)
        print("VAL", method, "macro-F1", round(row["macro_f1"], 4), "recall_1", round(row["recall_label_1"], 4))
    best_nb_method = _select_best(validation_rows, [NB_BASELINE, *NB_CANDIDATES])
    best_svm_method = _select_best(validation_rows, [SVM_BASELINE, *SVM_CANDIDATES])
    validation = pd.DataFrame(validation_rows).sort_values(
        ["family", "macro_f1", "recall_label_1", "method"], ascending=[True, False, False, True]
    )
    validation.to_csv(results_dir / "validation_metrics.csv", index=False, encoding="utf-8-sig")
    print("Selected NB:", best_nb_method)
    print("Selected SVM:", best_svm_method)

    final_methods = {"nb_word_1": NB_BASELINE, "nb_optimized": best_nb_method, "svm": best_svm_method}
    test_rows = []
    predictions = {}
    models = {}
    model_artifacts = {}
    model_configurations = {}
    for artifact_name, method in final_methods.items():
        started = time.perf_counter()
        model = make_pipeline(method).fit(train["message"], train["label"])
        fit_seconds = time.perf_counter() - started
        prediction = model.predict(test["message"])
        models[artifact_name] = model
        predictions[artifact_name] = prediction
        spec = get_model_spec(method)
        row = metric_row(artifact_name, "test", test["label"], prediction, spec["family"], method, fit_seconds)
        row.update(spec)
        test_rows.append(row)
        artifact_path = model_dir / f"{artifact_name}.joblib"
        joblib.dump(model, artifact_path)
        model_artifacts[artifact_name] = artifact_path.name
        model_configurations[artifact_name] = {**spec, "method": method, "artifact": artifact_path.name}
        print("TEST", artifact_name, row)

    metrics = pd.DataFrame(test_rows)
    metrics.to_csv(results_dir / "test_metrics.csv", index=False, encoding="utf-8-sig")
    metrics.to_csv(results_dir / "model_comparison.csv", index=False, encoding="utf-8-sig")

    output = test.rename(columns={"label": "true_label"}).copy()
    for artifact_name, prediction in predictions.items():
        output[f"prediction_{artifact_name}"] = prediction
    output.to_csv(results_dir / "test_predictions.csv", index=False, encoding="utf-8-sig")

    misclassified_rows = []
    for model_name, prediction in predictions.items():
        wrong = prediction != test["label"].to_numpy()
        for message, true_label, predicted_label in zip(test.loc[wrong, "message"], test.loc[wrong, "label"], prediction[wrong]):
            misclassified_rows.append({
                "model": model_name,
                "message_redacted": redact_message(message),
                "true_label": int(true_label),
                "predicted_label": int(predicted_label),
            })
    pd.DataFrame(misclassified_rows, columns=["model", "message_redacted", "true_label", "predicted_label"]).to_csv(
        results_dir / "misclassified_samples.csv", index=False, encoding="utf-8-sig"
    )

    prediction_columns = [f"prediction_{name}" for name in predictions]
    wrong_any = output.apply(
        lambda row: any(int(row[column]) != int(row["true_label"]) for column in prediction_columns), axis=1
    )
    error_output = output.loc[wrong_any].copy()
    error_output.insert(0, "message_redacted", error_output.pop("message").map(redact_message))
    error_output["wrong_models"] = error_output.apply(
        lambda row: ", ".join(name for name in predictions if int(row[f"prediction_{name}"]) != int(row["true_label"])), axis=1
    )
    error_output.to_csv(results_dir / "error_analysis.csv", index=False, encoding="utf-8-sig")
    error_output.head(10).to_csv(results_dir / "error_examples.csv", index=False, encoding="utf-8-sig")

    train.to_csv(processed_dir / "clean_train.csv", index=False, encoding="utf-8-sig")
    test.to_csv(processed_dir / "clean_test.csv", index=False, encoding="utf-8-sig")
    quality = {
        "train_quality": train_quality,
        "test_quality": test_quality,
        "class_counts_after_cleaning": {
            "train": {str(k): int(v) for k, v in train["label"].value_counts().sort_index().items()},
            "test": {str(k): int(v) for k, v in test["label"].value_counts().sort_index().items()},
        },
        "exact_cross_split_overlap_unique": overlap_unique,
        "exact_cross_split_train_rows_removed": overlap_rows,
        "train_after_overlap_check": len(train),
        "test_final": len(test),
        "preprocessing": {
            "unicode": "NFC",
            "case": "lowercase",
            "diacritics": "preserved in text; TF-IDF uses Unicode accent folding",
            "teencode": "conservative mixed-token mapping for @, 0, 4, 3, $, !",
            "urls_otp_money_phone_verification_terms": "preserved as model-visible text features",
            "same_transform_in_train_and_predict": True,
        },
        "leakage_controls": {
            "exact_normalized_cross_split_overlap_removed_from_train": True,
            "vectorizer_fit_scope": "train-fit for validation; full cleaned train for final models",
            "test_used_for_selection": False,
            "robustness_cases_used_for_training": False,
        },
    }
    (results_dir / "quality_report.json").write_text(json.dumps(quality, indent=2, ensure_ascii=False), encoding="utf-8")

    _save_feature_explanations(models, model_dir, results_dir)
    _save_robustness(models, results_dir)
    _save_figures(train, metrics, predictions, test["label"].to_numpy(), results_dir)

    best_overall_method = _select_best(validation_rows, CANDIDATES)
    metadata = {
        "model_type": "Binary Vietnamese SMS ham vs spam/scam",
        "labels": LABEL_NAMES,
        "selected_nb": "nb_optimized",
        "selected_nb_method": best_nb_method,
        "selected_svm": "svm",
        "selected_svm_method": best_svm_method,
        "recommended_model": "svm",
        "recommended_method": best_overall_method,
        "selected_by": "Macro-F1 validation, then recall and precision of label 1; test is held out",
        "models_compared_on_test": list(final_methods),
        "model_artifacts": model_artifacts,
        "model_configurations": model_configurations,
        "random_seed": SEED,
        "validation_size": 0.2,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "sklearn_version": sklearn.__version__,
        "dataset_version": {"train_sha256": sha256_file(train_file), "test_sha256": sha256_file(test_file)},
        "preprocessing": "NFC + lowercase + whitespace normalization + conservative teencode mapping in serialized pipeline",
        "explainability": "observed TF-IDF contribution; SVM margin coefficients or NB log-probability differences; not causal proof",
        "warning": "Label 1 combines spam and suspected scam; it is not proof of fraudulent behavior. SVM margins are not probabilities.",
    }
    (model_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Saved artifacts in", results_dir, "and", model_dir)
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-file", type=Path, default=RAW_DIR / "train.csv")
    parser.add_argument("--test-file", type=Path, default=RAW_DIR / "test.csv")
    parser.add_argument("--models-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--processed-dir", type=Path, default=PROCESSED_DIR)
    args = parser.parse_args()
    for path in [args.train_file, args.test_file]:
        if not path.exists():
            parser.error(f"Thiếu {path}. Hãy chạy: python download_data.py")
    run(args.train_file, args.test_file, args.models_dir, args.results_dir, args.processed_dir)


if __name__ == "__main__":
    main()
