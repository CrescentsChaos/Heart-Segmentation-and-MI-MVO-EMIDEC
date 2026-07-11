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

# Ablation variants ? display names (methodology Table 4.5)
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

# Published EMIDEC / LGE-CMR comparison targets (methodology Table 4.7 + 2025 papers)
SOTA_BENCHMARKS = [
    {
        "method": "nnU-Net",
        "citation": "Isensee et al.",
        "year": 2021,
        "architecture": "Self-configured 3D U-Net",
        "LV": 0.941,
        "MYO": 0.856,
        "MI": 0.720,
        "MVO": None,
        "dataset": "EMIDEC",
    },
    {
        "method": "ICPIU-Net",
        "citation": "Brahim et al.",
        "year": 2022,
        "architecture": "Cascaded 3D + VAE prior",
        "LV": 0.932,
        "MYO": 0.895,
        "MI": 0.783,
        "MVO": None,
        "dataset": "EMIDEC",
    },
    {
        "method": "2D-3D Cascade",
        "citation": "Schwab et al.",
        "year": 2025,
        "architecture": "Error-correcting 2D-3D cascade CNN",
        "LV": None,
        "MYO": 0.830,
        "MI": 0.720,
        "MVO": None,
        "dataset": "LGE-CMR / EMIDEC-related",
    },
]

# Primary success criterion from revised methodology
TARGET_MI_DICE = 0.783  # beat ICPIU-Net
