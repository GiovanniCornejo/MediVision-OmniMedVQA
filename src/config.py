import os

# ------------------------------- Dataset Paths ------------------------------ #
DEFAULT_DATA_DIR = os.path.join(".", "data", "OmniMedVQA")
DATA_DIR = os.environ.get("OMNIMEDVQA_DATA_DIR", DEFAULT_DATA_DIR) # Allow overriding via environment variable
QA_DIR = os.path.join(DATA_DIR, "QA_information", "Open-access")
IMG_DIR = os.path.join(DATA_DIR, "Images")

# ---------------------------- Label Mapping Paths --------------------------- #
LABEL_MAPPING_DIR = os.path.join("data", "label_mappings")

NO_FINDING_MAP = os.path.join(LABEL_MAPPING_DIR, "no_finding_map.csv")
INCONCLUSIVE_MAP = os.path.join(LABEL_MAPPING_DIR, "inconclusive_map.csv")
YES_FINDING_MAP = os.path.join(LABEL_MAPPING_DIR, "yes_finding_map.csv")

# --------------------------- Question Type Filter --------------------------- #
QUESTION_TYPE = "Disease Diagnosis"

# ------------------------------ Option Columns ------------------------------ #
OPTION_COLS = ["option_A", "option_B", "option_C", "option_D"]

# ----------------------- Train/Validation/Test Splits ----------------------- #
# Reserve 10% for held-out test set
TEST_SPLIT_RATIO = 0.10

# Remaining 90% split into 85% training / 15% validation
TRAIN_VAL_RATIO = 0.85

# ---------------------- Random Seed for Reproducibility --------------------- #
SEED = 42