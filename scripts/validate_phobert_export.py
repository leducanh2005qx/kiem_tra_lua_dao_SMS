"""Validate and optionally run a CPU smoke test for an exported PhoBERT model."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REQUIRED_FILES = {"config.json", "label_map.json", "preprocessing_config.json", "metrics.json", "model_manifest.json"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir")
    parser.add_argument("--text", default="")
    args = parser.parse_args()
    model_dir = Path(args.model_dir)
    missing = sorted(name for name in REQUIRED_FILES if not (model_dir / name).exists())
    weight_exists = any((model_dir / name).exists() for name in ("model.safetensors", "pytorch_model.bin"))
    tokenizer_exists = (model_dir / "tokenizer.json").exists() or (model_dir / "sentencepiece.bpe.model").exists()
    result = {"model_dir": str(model_dir), "missing_metadata": missing, "weights_present": weight_exists, "tokenizer_present": tokenizer_exists, "valid": not missing and weight_exists and tokenizer_exists}
    if result["valid"] and args.text:
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
            model = AutoModelForSequenceClassification.from_pretrained(model_dir, local_files_only=True).to("cpu")
            inputs = tokenizer(args.text, return_tensors="pt", truncation=True, max_length=256)
            with torch.no_grad():
                logits = model(**inputs).logits
            result["cpu_prediction_id"] = int(logits.argmax(dim=-1).item())
            result["cpu_smoke_test"] = True
        except Exception as exc:  # pragma: no cover - optional dependency path
            result["cpu_smoke_test"] = False
            result["cpu_error"] = str(exc)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result["valid"] else 1)


if __name__ == "__main__":
    main()
