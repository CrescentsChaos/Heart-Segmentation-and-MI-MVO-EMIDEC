"""Official model naming for thesis / paper comparison."""

# Full proposed model (ablation M5)
MODEL_NAME = "AFDD-Net"
MODEL_FULL_NAME = (
    "Anisotropic Factorized Dual-Decoder Network "
    "with MYO Soft-Gating and Topology Consistency"
)
MODEL_YEAR = 2026

# Short citation-style string used in figures and tables
MODEL_CITE = f"{MODEL_NAME} (this work)"

# Ablation variants — display names (methodology Table 4.5)
VARIANT_NAMES = {
    "M1": "3D U-Net baseline",
    "M2": f"{MODEL_NAME}-F (factorized conv)",
    "M3": f"{MODEL_NAME}-D (dual decoder)",
    "M4": f"{MODEL_NAME}-T (Focal Tversky)",
    "M5": f"{MODEL_NAME} (full proposed)",
}

VARIANT_SHORT = {
    "M1": "Baseline 3D U-Net",
    "M2": f"{MODEL_NAME}-F",
    "M3": f"{MODEL_NAME}-D",
    "M4": f"{MODEL_NAME}-T",
    "M5": MODEL_NAME,
}

# Verified EMIDEC-only MI/scar Dice comparators (methodology Table 4.7).
# Removed Isensee et al. 2021 nnU-Net (private LGE cohort, not EMIDEC) and
# non-EMIDEC 2025–2026 papers. Protocols differ; do not mix without noting.
SOTA_BENCHMARKS = [
    {
        "method": "Zhang (cascaded nnU-Net)",
        "citation": "Zhang et al.",
        "year": 2021,
        "architecture": "2D-3D cascaded nnU-Net",
        "LV": None,
        "MYO": 0.879,
        "MI": 0.712,
        "MVO": 0.785,
        "dataset": "EMIDEC",
        "protocol": "Official test (50 cases)",
    },
    {
        "method": "ICPIU-Net",
        "citation": "Brahim et al.",
        "year": 2022,
        "architecture": "Cascaded 3D + VAE prior",
        "LV": None,
        "MYO": 0.877,
        "MI": 0.734,
        "MVO": None,
        "dataset": "EMIDEC",
        "protocol": "Official test (50 cases)",
    },
    {
        "method": "ICPIU-Net (5-fold)",
        "citation": "Brahim et al.",
        "year": 2022,
        "architecture": "Cascaded 3D + VAE prior",
        "LV": 0.932,
        "MYO": 0.895,
        "MI": 0.783,
        "MVO": None,
        "dataset": "EMIDEC",
        "protocol": "5-fold CV (100 cases)",
    },
    {
        "method": "3D nnU-Net (EMIDEC)",
        "citation": "nnU-Net baseline",
        "year": 2021,
        "architecture": "nnU-Net 3D",
        "LV": None,
        "MYO": 0.872,
        "MI": 0.688,
        "MVO": None,
        "dataset": "EMIDEC",
        "protocol": "5-fold CV",
    },
    {
        "method": "2D nnU-Net (EMIDEC)",
        "citation": "nnU-Net baseline",
        "year": 2021,
        "architecture": "nnU-Net 2D",
        "LV": None,
        "MYO": 0.851,
        "MI": 0.509,
        "MVO": None,
        "dataset": "EMIDEC",
        "protocol": "5-fold CV",
    },
    {
        "method": "GAN-aug. cascade",
        "citation": "Lustermans et al.",
        "year": 2022,
        "architecture": "Cascaded nnU-Net + cGAN",
        "LV": None,
        "MYO": 0.840,
        "MI": 0.720,
        "MVO": None,
        "dataset": "EMIDEC",
        "protocol": "Test / per-slice (see paper)",
    },
    {
        "method": "CLAIM",
        "citation": "Ramzan et al.",
        "year": 2025,
        "architecture": "Diffusion augmentation + nnU-Net",
        "LV": None,
        "MYO": None,
        "MI": 0.635,
        "MVO": None,
        "dataset": "EMIDEC",
        "protocol": "10 held-out cases",
    },
    {
        "method": "2D-3D Cascade (EcorC)",
        "citation": "Schwab et al.",
        "year": 2025,
        "architecture": "Error-correcting 2D-3D cascade CNN",
        "LV": None,
        "MYO": 0.860,
        "MI": 0.760,
        "MVO": None,
        "dataset": "EMIDEC",
        "protocol": "5-fold CV (100 cases)",
    },
    {
        "method": "Expert (inter-observer)",
        "citation": "Lalande et al.",
        "year": 2020,
        "architecture": "Human expert ceiling",
        "LV": None,
        "MYO": 0.830,
        "MI": 0.690,
        "MVO": None,
        "dataset": "EMIDEC",
        "protocol": "Inter-observer (Data 2020)",
    },
]

# Beat current published EMIDEC best (Schwab 2025, 5-fold CV).
# ICPIU-Net 5-fold MI Dice 0.783 remains the stretch target in the table above.
TARGET_MI_DICE = 0.760
TARGET_MI_DICE_LABEL = "Schwab 2025 (EMIDEC best)"
