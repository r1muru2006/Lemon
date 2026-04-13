from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
FLOWS_DIR = DATA_DIR / "flows"
FEATURES_DIR = DATA_DIR / "features"
MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"

EPS = 1e-9

TIMESTAMP_CANDIDATES = [
    "timestamp",
    "flow_start_time",
    "flow_end_time",
    "event_time",
    "datetime",
    "time",
]

LABEL_CANDIDATES = [
    "label",
    "class",
    "attack",
    "attack_type",
    "category",
]

ATTACK_FAMILY_KEYWORDS = {
    "syn": "syn_flood",
    "udp": "udp_flood",
    "dns": "dns_flood",
    "ntp": "ntp_flood",
    "ldap": "ldap_flood",
    "http": "http_flood",
    "icmp": "icmp_flood",
}

LEAKY_COLUMNS = {
    "flow_id",
    "id",
}

TRAIN_FILE = "train.parquet"
VAL_FILE = "val.parquet"
TEST_FILE = "test.parquet"
MODEL_MAIN_FILE = "main.pkl"
MODEL_BASELINE_FILE = "baseline.pkl"
METADATA_FILE = "metadata.json"

DEFAULT_MAX_FPR = 0.01
