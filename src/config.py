from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
FIGURE_DIR = RESULTS_DIR / "figures"
APP_HISTORY_DB = ROOT / "data" / "app_history.db"
ANNOTATION_QUEUE = ROOT / "data" / "annotations" / "annotation_queue.csv"
ANNOTATION_LABEL_DB = ROOT / "data" / "annotation_labels.db"
MULTICLASS_ANNOTATIONS = ROOT / "data" / "multiclass" / "annotations.csv"

DATASET_REPO = "https://huggingface.co/datasets/trannguyenthaituan/vietnamese_sms_dataset"
DATA_URLS = {
    "train.csv": f"{DATASET_REPO}/resolve/main/train.csv",
    "test.csv": f"{DATASET_REPO}/resolve/main/test.csv",
}
LABEL_NAMES = {0: "SMS hợp lệ", 1: "SMS rác / nghi vấn lừa đảo"}
MULTICLASS_LABEL_NAMES = {0: "LEGITIMATE", 1: "PROMOTION", 2: "PHISHING", 3: "FINANCIAL_SCAM"}
SEED = 42
