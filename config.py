"""Configuration for the revised EMIDEC dual-decoder methodology."""
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths

# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
EMIDEC_ROOT = Path(r"E:\emidec-dataset-1.0.1")
DATASET_DIR = ROOT / "Dataset"
CHECKPOINT_DIR = ROOT / "checkpoints"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"

# ---------------------------------------------------------------------------
# Volume geometry (EMIDEC anisotropic MRI)

# ---------------------------------------------------------------------------
# In-plane ~1.5 mm, through-plane ~10 mm (methodology Sec. 4.2)
TARGET_SPACING = (1.5, 1.5, 10.0)
# Depth padded/resized to 16 so four stride-2 pools reach a valid bottleneck
TARGET_SHAPE = (128, 128, 16)  # (H, W, D)

# ---------------------------------------------------------------------------
# Label protocol (EMIDEC official)
#   0 = background
#   1 = LV cavity
#   2 = normal myocardium
#   3 = myocardial infarction (MI)
#   4 = no-reflow / MVO
# RV is NOT annotated in EMIDEC ? anatomy uses BG / LV / MYO only.
# Full myocardial wall for gating = labels {2, 3, 4}.

# ---------------------------------------------------------------------------
RAW_BG, RAW_LV, RAW_MYO, RAW_MI, RAW_MVO = 0, 1, 2, 3, 4
ANAT_BG, ANAT_LV, ANAT_MYO = 0, 1, 2
NUM_ANATOMY_CLASSES = 3  # BG, LV, MYO  (RV omitted - no EMIDEC mask)
NUM_PATHOLOGY_CLASSES = 2  # MI, MVO (independent sigmoids)
ANATOMY_CLASS_NAMES = ["BG", "LV", "MYO"]
PATHOLOGY_CLASS_NAMES = ["MI", "MVO"]
# Anatomy CE weights (methodology Sec. 4.4.1; RV weight dropped)
ANATOMY_CE_WEIGHTS = [0.1, 1.0, 1.5]  # BG, LV, MYO

# ---------------------------------------------------------------------------
# Model / training (methodology Sec. 4.5)

# ---------------------------------------------------------------------------
IN_CHANNELS = 1
BASE_FILTERS = [32, 64, 128, 256]
USE_FACTORIZED = True
LR = 1e-4
MIN_LR = 1e-6
EPOCHS = 150
BATCH_SIZE = 2
NUM_WORKERS = 0
LAMBDA_FTL = 1.0
LAMBDA_TOPO = 0.5
# Focal Tversky (Sec. 4.4.2)
FTL_ALPHA = 0.7
FTL_BETA = 0.3
FTL_GAMMA = 0.75
FTL_EPS = 1e-5
SEED = 42
DEVICE = "cuda"  # overridden at runtime if CUDA unavailable
# Ablation variant keys: M1 .. M5 (Sec. 4.5 Table)
DEFAULT_VARIANT = "M5"
