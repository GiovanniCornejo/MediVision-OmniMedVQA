import os
import json
import pandas as pd
from datasets import load_dataset, Dataset
from sklearn.model_selection import train_test_split

from .config import (
    QA_DIR, LABEL_MAPPING_DIR, NO_FINDING_MAP, INCONCLUSIVE_MAP, YES_FINDING_MAP,
    QUESTION_TYPE, OPTION_COLS, SEED, TEST_SPLIT_RATIO, TRAIN_VAL_RATIO
)

# ---------------------------------------------------------------------------- #
#                               Helper Functions                               #
# ---------------------------------------------------------------------------- #

def list_json_files(qa_dir=QA_DIR):
    """Return list of JSON files in the QA directory."""
    return [os.path.join(qa_dir, f) for f in os.listdir(qa_dir) if f.endswith(".json")]

def load_dataset_df(json_files):
    """Load JSON files into a Hugging Face dataset and convert to pandas DataFrame."""
    dataset: Dataset = load_dataset("json", data_files=json_files, split="train") # type: ignore
    return dataset.to_pandas()

def add_gt_label(df):
    """Add 'gt_label' column indicating which option matches 'gt_answer'."""
    def find_correct_option(row):
        for col in OPTION_COLS:
            if str(row[col]).strip().lower().rstrip(".") == str(row["gt_answer"]).strip().lower().rstrip("."):
                return col
        return None
    df["gt_label"] = df.apply(find_correct_option, axis=1)
    return df

# ---------------------------------------------------------------------------- #
#                            Data Cleaning Functions                           #
# ---------------------------------------------------------------------------- #

def fix_schema(json_files):
    """
    Fix known schema issues in JSON files.
    """
    for f in json_files:
        with open(f, "r", encoding="utf-8") as file:
            data = json.load(file)
        
        modified = False
        for entry in data:
            if "modality" in entry:
                entry["modality_type"] = entry.pop("modality")
                modified = True
        
        if modified:
            print(f"Fixed schema in {os.path.basename(f)}")
            with open(f, "w", encoding="utf-8") as file:
                json.dump(data, file, indent=2)

def normalize_text(s: str) -> str:
    """Standardize text for matching against mapping files."""
    return str(s).strip().lower().strip('"').rstrip(".")

def apply_single_label_mapping(df: pd.DataFrame, mapping_csv_path, unified_label) -> pd.DataFrame:
    """
    Apply label normalization using a CSV mapping file.
    - mapping_csv_path must have a column 'raw_label' listing variants.
    - Replaces any match in option_* with unified_label.
    """
    mapping_df = pd.read_csv(mapping_csv_path)
    mapping_dict = {normalize_text(raw): unified_label for raw in mapping_df["raw_label"]}

    for col in OPTION_COLS:
        df[col] = df[col].apply(lambda x: mapping_dict.get(normalize_text(x), x))
    return df

def preprocess_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all label preprocessing steps:
    - Convert 'No' style answers to 'No Finding'
    - Convert inconclusive answers to 'Inconclusive'
    - Convert generic 'Yes/Abnormal' answers to 'Abnormal (unspecified)'
    """

    # Step 1: No Finding
    df = apply_single_label_mapping(
        df, NO_FINDING_MAP, "No Finding"
    )

    # Step 2: Inconclusive
    df = apply_single_label_mapping(
        df, INCONCLUSIVE_MAP, "Inconclusive"
    )

    # Step 3: Abnormal (unspecified)
    df = apply_single_label_mapping(
        df, YES_FINDING_MAP, "Abnormal (unspecified)"
    )

    return df

def reduce_duplicates(row) -> pd.Series:
    gt_key = row["gt_label"]
    gt_text = row[gt_key]
    opts = [row[col] for col in OPTION_COLS]
    seen = {}
    for opt in opts:
        if opt not in seen:
            seen[opt] = True
    unique_opts = list(seen.keys())
    if gt_text not in unique_opts:
        unique_opts.append(gt_text)
    while len(unique_opts) < len(OPTION_COLS):
        unique_opts.append(None)
    unique_opts = unique_opts[:len(OPTION_COLS)]
    for col, val in zip(OPTION_COLS, unique_opts):
        row[col] = val
    # recompute gt_label
    for col in OPTION_COLS:
        if row[col] == gt_text:
            row["gt_label"] = col
            break
    return row


# ---------------------------------------------------------------------------- #
#                                 Main Function                                #
# ---------------------------------------------------------------------------- #

def load_omnimed_dataset(
        qa_dir=QA_DIR,
        test_ratio=TEST_SPLIT_RATIO,
        train_val_ratio=TRAIN_VAL_RATIO,
        seed=SEED,
        stratify_col="modality_type",
        apply_label_normalization=True
):
    """
    Load OmniMedVQA dataset, fix schema, and return train/validation/test splits as DataFrames.
    Splitting is done by unique images, so all QA pairs for the same image stay together.

    Args:
        qa_dir (str): 
            path to QA JSON files
        test_ratio (float): 
            fraction of data reserved for test
        train_val_ratio (float): 
            fraction of remaining data used for training
        seed (int): 
            random seed
        stratify_col (str, optional): 
            column name to stratify by (e.g., `modality_type`)

    Returns:
        train_df, val_df, test_df (pd.DataFrame)
    """
    # List all JSON files
    json_files = list_json_files(qa_dir)

    # Fix schema issues
    fix_schema(json_files)

    # Load unified DataFrame
    df: pd.DataFrame = load_dataset_df(json_files) # type: ignore

    # Filter to only Disease Diagnosis if parameter is set
    df = df[df['question_type'] == QUESTION_TYPE].reset_index(drop=True)

    # Replace gt_answer text with gt_label mapping
    df = add_gt_label(df)
    # df.drop(columns=["gt_answer"], inplace=True)

    # Apply preprocessing (label normalization)
    if apply_label_normalization:
        df = preprocess_labels(df)

    # Reduce duplicate options
    df = df.apply(reduce_duplicates, axis=1)

    # Get unique images
    unique_images = df['image_path'].unique()

    if stratify_col and stratify_col in df.columns:
        # Compute stratification label per image (take first QA's value for simplicity)
        image_labels = df.groupby('image_path')[stratify_col].first().loc[unique_images]

        # Split off hold-out test set
        train_val_images, test_images = train_test_split(
            unique_images,
            test_size=test_ratio,
            random_state=seed,
            stratify=image_labels
        )

        # Split further into training and validation sets
        train_val_labels = image_labels.loc[train_val_images]
        train_images, val_images = train_test_split(
            train_val_images,
            train_size=train_val_ratio,
            random_state=seed,
            stratify=train_val_labels
        )
    else:
        # If no stratification column, do simple random split
        train_val_images, test_images = train_test_split(
            unique_images,
            test_size=test_ratio,
            random_state=seed
        )
        train_images, val_images = train_test_split(
            train_val_images,
            train_size=train_val_ratio,
            random_state=seed
        )

    # Map back to QA items
    train_df = df[df['image_path'].isin(train_images)].reset_index(drop=True)
    val_df = df[df['image_path'].isin(val_images)].reset_index(drop=True)
    test_df = df[df['image_path'].isin(test_images)].reset_index(drop=True)

    return train_df, val_df, test_df
