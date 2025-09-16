import os
import json
import pandas as pd
from datasets import load_dataset, Dataset
from sklearn.model_selection import train_test_split

from .config import QA_DIR, SEED, TEST_SPLIT_RATIO, TRAIN_VAL_RATIO

# ---------------------------------------------------------------------------- #
#                               Helper Functions                               #
# ---------------------------------------------------------------------------- #

def list_json_files(qa_dir=QA_DIR):
    """Return list of JSON files in the QA directory."""
    return [os.path.join(qa_dir, f) for f in os.listdir(qa_dir) if f.endswith(".json")]

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

def load_dataset_df(json_files):
    "Load JSON files into a Hugging Face dataset and convert to pandas DataFrame."
    dataset: Dataset = load_dataset("json", data_files=json_files, split="train") # type: ignore
    return dataset.to_pandas()

# ---------------------------------------------------------------------------- #
#                                 Main Function                                #
# ---------------------------------------------------------------------------- #

def load_omnimed_dataset(
        qa_dir=QA_DIR,
        test_ratio=TEST_SPLIT_RATIO,
        train_val_ratio=TRAIN_VAL_RATIO,
        seed=SEED,
        stratify_col="modality_type"
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
