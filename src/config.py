from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
FIGURE_DIR = RESULTS_DIR / "figures"
APP_HISTORY_DB = ROOT / "data" / "app_history.db"
MULTICLASS_ANNOTATIONS = ROOT / "data" / "multiclass" / "annotations.csv"

DATASET_REPO = "https://huggingface.co/datasets/trannguyenthaituan/vietnamese_sms_dataset"
DATA_URLS = {
    "train.csv": f"{DATASET_REPO}/resolve/main/train.csv",
    "test.csv": f"{DATASET_REPO}/resolve/main/test.csv",
}
LABEL_NAMES = {0: "SMS hợp lệ", 1: "SMS rác / nghi vấn lừa đảo"}
SEED = 42
