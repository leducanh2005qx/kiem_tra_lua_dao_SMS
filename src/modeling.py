"""Leakage-safe model specifications, pipelines and local explanations."""
from copy import deepcopy

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.preprocessing import FunctionTransformer

from src.preprocess import normalize_message, normalize_messages


NB_ALPHAS = (0.1, 0.3, 0.5, 1.0, 2.0)
NB_FIT_PRIORS = (True, False)
SVM_C_VALUES = (0.1, 0.5, 1.0, 2.0)
SVM_CLASS_WEIGHTS = (None, "balanced")
VECTORIZER_SPECS = {
    "word_1": {"analyzer": "word", "ngram_range": (1, 1)},
    "word_12": {"analyzer": "word", "ngram_range": (1, 2)},
    "char_35": {"analyzer": "char", "ngram_range": (3, 5)},
}


def _number_text(value):
    return str(value).replace(".", "p")


def nb_candidate_name(vectorizer, alpha, fit_prior):
    """Return a filesystem-safe, explicit name for an NB experiment."""
    prior_text = "true" if fit_prior else "false"
    return f"nb_{vectorizer}_a{_number_text(alpha)}_prior_{prior_text}"


def svm_candidate_name(vectorizer, c_value, class_weight):
    """Return a filesystem-safe, explicit name for an SVM experiment."""
    weight_text = "balanced" if class_weight == "balanced" else "none"
    return f"svm_{vectorizer}_c{_number_text(c_value)}_weight_{weight_text}"


NB_CANDIDATES = tuple(
    nb_candidate_name(vectorizer, alpha, fit_prior)
    for vectorizer in VECTORIZER_SPECS
    for alpha in NB_ALPHAS
    for fit_prior in NB_FIT_PRIORS
)
SVM_CANDIDATES = tuple(
    svm_candidate_name(vectorizer, c_value, class_weight)
    for vectorizer in VECTORIZER_SPECS
    for c_value in SVM_C_VALUES
    for class_weight in SVM_CLASS_WEIGHTS
)

NB_BASELINE = "nb_word_1"
SVM_BASELINE = "svm_word_12"


MODEL_SPECS = {
    NB_BASELINE: {
        "family": "Multinomial Naive Bayes",
        "vectorizer": "word_1",
        "alpha": 1.0,
        "fit_prior": True,
        "role": "baseline",
    },
    SVM_BASELINE: {
        "family": "Linear SVM",
        "vectorizer": "word_12",
        "C": 1.0,
        "class_weight": None,
        "role": "baseline",
    },
    # Stable aliases retained for callers of the original project.
    "svm_word_1": {
        "family": "Linear SVM",
        "vectorizer": "word_1",
        "C": 1.0,
        "class_weight": None,
        "role": "baseline_alias",
    },
    "svm_char_35": {
        "family": "Linear SVM",
        "vectorizer": "char_35",
        "C": 1.0,
        "class_weight": None,
        "role": "baseline_alias",
    },
}

for _candidate in NB_CANDIDATES:
    _parts = _candidate.split("_")
    _vectorizer = f"{_parts[1]}_{_parts[2]}"
    MODEL_SPECS[_candidate] = {
        "family": "Multinomial Naive Bayes",
        "vectorizer": _vectorizer,
        "alpha": float(_parts[3][1:].replace("p", ".")),
        "fit_prior": _parts[5] == "true",
        "role": "candidate",
    }

for _candidate in SVM_CANDIDATES:
    _parts = _candidate.split("_")
    _vectorizer = f"{_parts[1]}_{_parts[2]}"
    MODEL_SPECS[_candidate] = {
        "family": "Linear SVM",
        "vectorizer": _vectorizer,
        "C": float(_parts[3][1:].replace("p", ".")),
        "class_weight": "balanced" if _parts[5] == "balanced" else None,
        "role": "candidate",
    }


def get_model_spec(method):
    """Return a copy so callers cannot mutate the registry."""
    if method not in MODEL_SPECS:
        raise ValueError(f"Unknown method: {method}")
    return deepcopy(MODEL_SPECS[method])


def make_pipeline(method):
    """Build a complete serializable preprocessing + TF-IDF + classifier pipeline."""
    spec = get_model_spec(method)
    vectorizer_args = {
        **VECTORIZER_SPECS[spec["vectorizer"]],
        "min_df": 1,
        "sublinear_tf": True,
        "strip_accents": "unicode",
    }
    if vectorizer_args["analyzer"] == "word":
        vectorizer_args["token_pattern"] = r"(?u)\b\w[\w@!#$]*\b"
    vectorizer = TfidfVectorizer(**vectorizer_args)
    if spec["family"] == "Multinomial Naive Bayes":
        classifier = MultinomialNB(alpha=spec["alpha"], fit_prior=spec["fit_prior"])
    else:
        classifier = LinearSVC(
            C=spec["C"],
            class_weight=spec["class_weight"],
            random_state=42,
        )
    return Pipeline([
        ("normalize", FunctionTransformer(normalize_messages, validate=False)),
        ("tfidf", vectorizer),
        ("clf", classifier),
    ])


def _feature_weights(model):
    clf = model.named_steps["clf"]
    if hasattr(clf, "coef_"):
        return np.asarray(clf.coef_[0]), "linear margin"
    return np.asarray(clf.feature_log_prob_[1] - clf.feature_log_prob_[0]), "log-probability difference"


def explain_message(model, message, n=10):
    """Return observed feature contributions for both target labels.

    These are model-attribution scores, not causal or human-verifiable proof.
    Positive scores support label 1; negative scores support label 0.
    """
    if n < 1:
        return []
    tfidf = model.named_steps["tfidf"]
    x = tfidf.transform([normalize_message(message)]).tocoo()
    names = tfidf.get_feature_names_out()
    weights, method = _feature_weights(model)
    contributions = [
        (str(names[int(idx)]), float(value * weights[int(idx)]))
        for idx, value in zip(x.col, x.data)
    ]
    positive = sorted(
        ((term, score) for term, score in contributions if score > 0),
        key=lambda item: item[1],
        reverse=True,
    )[:n]
    negative = sorted(
        ((term, score) for term, score in contributions if score < 0),
        key=lambda item: item[1],
    )[:n]
    rows = [
        {"feature": term, "contribution": round(score, 6), "toward_label": 1, "method": method}
        for term, score in positive
    ]
    rows.extend(
        {"feature": term, "contribution": round(score, 6), "toward_label": 0, "method": method}
        for term, score in negative
    )
    prediction = int(model.predict([message])[0])
    for row in rows:
        row["predicted_label"] = prediction
        row["relevant_to_prediction"] = row["toward_label"] == prediction
    return rows


def get_evidence(model, message, n=8):
    """Backward-compatible compact evidence list for the Streamlit page."""
    rows = explain_message(model, message, n=n)
    prediction = int(model.predict([message])[0])
    return [
        (row["feature"], row["contribution"])
        for row in rows
        if row["toward_label"] == prediction
    ][:n]
