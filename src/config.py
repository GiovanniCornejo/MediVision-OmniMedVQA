import os

# ------------------------------- Dataset Paths ------------------------------ #
DEFAULT_DATA_DIR = os.path.join(".", "data", "OmniMedVQA")
DATA_DIR = os.environ.get("OMNIMEDVQA_DATA_DIR", DEFAULT_DATA_DIR) # Allow overriding via environment variable
QA_DIR = os.path.join(DATA_DIR, "QA_information", "Open-access")
IMG_DIR = os.path.join(DATA_DIR, "Images")

# ----------------------- Train/Validation/Test Splits ----------------------- #
# Reserve 10% for held-out test set
TEST_SPLIT_RATIO = 0.10

# Remaining 90% split into 85% training / 15% validation
TRAIN_VAL_RATIO = 0.85

# --------------------------- Question Type Filter --------------------------- #
QUESTION_TYPE = "Disease Diagnosis"

# ---------------------- Random Seed for Reproducibility --------------------- #
SEED = 42