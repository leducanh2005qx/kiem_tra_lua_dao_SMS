"""Optional PhoBERT research path, isolated from the production TF-IDF models.

The main environment intentionally does not import torch/transformers.  Install
``requirements-transformers.txt`` only when a real fine-tuning experiment is
authorized and resources are available.
"""
from __future__ import annotations

import importlib.util
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from src.preprocess import normalize_message


PHOBERT_MODEL_ID = "vinai/phobert-base-v2"
PHOBERT_LICENSE = "AGPL-3.0 (verify compatibility before deployment)"


class OptionalDependencyError(RuntimeError):
    pass


@dataclass
class PhoBERTTrainingConfig:
    model_id: str = PHOBERT_MODEL_ID
    max_length: int = 256
    epochs: int = 3
    learning_rate: float = 2e-5
    train_batch_size: int = 8
    eval_batch_size: int = 8
    seed: int = 42
    early_stopping_patience: int = 2
    local_files_only: bool = False


def dependency_status() -> dict:
    status = {name: importlib.util.find_spec(name) is not None for name in ("torch", "transformers", "underthesea", "datasets")}
    cuda = False
    device = "cpu"
    if status["torch"]:
        import torch
        cuda = bool(torch.cuda.is_available())
        device = "cuda" if cuda else "cpu"
    return {**status, "cuda": cuda, "device": device, "model_id": PHOBERT_MODEL_ID, "license": PHOBERT_LICENSE}


def _require(*packages):
    missing = [package for package in packages if importlib.util.find_spec(package) is None]
    if missing:
        raise OptionalDependencyError(f"Thiếu dependency tùy chọn: {', '.join(missing)}. Cài requirements-transformers.txt.")


def segment_text(text: str) -> str:
    """Segment Vietnamese text as required by PhoBERT's tokenizer contract."""
    _require("underthesea")
    from underthesea import word_tokenize
    return word_tokenize(normalize_message(text), format="text")


def prepare_texts(texts):
    return [segment_text(text) for text in texts]


def load_phobert(model_id=PHOBERT_MODEL_ID, num_labels=2, local_files_only=False):
    """Load tokenizer and classifier lazily; never contacts a URL itself."""
    _require("torch", "transformers", "underthesea")
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    import torch
    tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=local_files_only)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id, num_labels=num_labels, local_files_only=local_files_only
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return {"tokenizer": tokenizer, "model": model, "device": str(device), "model_id": model_id}


def fine_tune_phobert(dataset_path, output_dir, label_mapping=None, config=None):
    """Fine-tune on a real, explicitly labeled CSV with early stopping.

    The CSV must contain ``message`` and ``label``.  This function is opt-in,
    requires the transformer dependencies, and does not touch production models.
    """
    _require("torch", "transformers", "underthesea", "datasets")
    import numpy as np
    import pandas as pd
    from datasets import Dataset
    from sklearn.model_selection import train_test_split
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        EarlyStoppingCallback,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    config = config or PhoBERTTrainingConfig()
    set_seed(config.seed)
    frame = pd.read_csv(dataset_path, encoding="utf-8-sig")
    if {"message", "label"} - set(frame.columns):
        raise ValueError("PhoBERT CSV cần cột message,label.")
    frame = frame.dropna(subset=["message", "label"]).copy()
    frame["message"] = prepare_texts(frame["message"].astype(str).tolist())
    mapping = label_mapping or {str(value): int(value) for value in sorted(frame["label"].unique())}
    frame["label_id"] = frame["label"].astype(str).map({str(key): value for key, value in mapping.items()})
    if frame["label_id"].isna().any():
        raise ValueError("Có label không nằm trong label_mapping đã khai báo.")
    train_frame, validation_frame = train_test_split(
        frame, test_size=0.2, random_state=config.seed, stratify=frame["label_id"]
    )
    tokenizer = AutoTokenizer.from_pretrained(config.model_id, local_files_only=config.local_files_only)

    def tokenize(batch):
        return tokenizer(batch["message"], truncation=True, max_length=config.max_length)

    train_dataset = Dataset.from_pandas(train_frame[["message", "label_id"]].rename(columns={"label_id": "labels"})).map(tokenize, batched=True)
    validation_dataset = Dataset.from_pandas(validation_frame[["message", "label_id"]].rename(columns={"label_id": "labels"})).map(tokenize, batched=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        config.model_id, num_labels=len(mapping), local_files_only=config.local_files_only
    )
    args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=config.epochs,
        learning_rate=config.learning_rate,
        per_device_train_batch_size=config.train_batch_size,
        per_device_eval_batch_size=config.eval_batch_size,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_strategy="epoch",
        report_to=[],
        seed=config.seed,
    )

    def compute_metrics(eval_prediction):
        logits, labels = eval_prediction
        predictions = np.argmax(logits, axis=-1)
        return {"macro_f1": float(f1_score(labels, predictions, average="macro"))}

    from sklearn.metrics import f1_score
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=config.early_stopping_patience)],
    )
    trainer.train()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    metadata = {"config": asdict(config), "label_mapping": mapping, "license": PHOBERT_LICENSE, "training_log": str(output_dir / "trainer_state.json")}
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    return metadata
