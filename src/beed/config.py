"""Central configuration — paths and dataset constants."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"

# Dataset constants
RAW_FILE = DATA_RAW / "BEED_Data.csv"
SAMPLING_RATE = 256          # Hz
N_CHANNELS = 16
N_UMAP_COMPONENTS = 3
RANDOM_STATE = 42

CHANNEL_NAMES = [
    "Fp1", "Fp2", "F3", "F4", "C3", "C4",
    "P3",  "P4",  "O1", "O2", "F7", "F8",
    "T3",  "T4",  "T5", "T6",
]

CLASS_LABELS = {
    0: "Healthy",
    1: "Focal",
    2: "Generalized",
    3: "Seizure Events",
}
