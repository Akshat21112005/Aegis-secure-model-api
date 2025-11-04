# --- data_pipeline.py ---
# This script builds the master engineered training dataset.
# 1. Finds all raw data files in the /data/ directory.
# 2. Samples a fraction (defined in config.py) from each file.
# 3. Appends them into a single DataFrame.
# 4. Runs the full feature extraction pipeline on them.
# 5. Saves the final engineered DataFrame to `engineered_features.csv`.

import os
import glob
import pandas as pd
import config
from feature_extraction import extract_features_from_dataframe

def load_and_sample_raw_data(data_dir, fraction=0.5, random_state=42):
    """
    Loads all .csv files from the data_dir, samples them, and appends.
    Assumes CSVs have 'label' and 'url' columns.
    """
    # This line does exactly what you want: finds all .csv files in the folder.
    raw_data_files = glob.glob(os.path.join(data_dir, "*.csv"))
    
    if not raw_data_files:
        print(f"Error: No .csv files found in '{data_dir}'.")
        print("Please add your raw data files (e.g., phishing.csv, legit.csv) to the /data/ folder.")
        return pd.DataFrame()

    print(f"Found {len(raw_data_files)} raw data files.")
    
    all_samples = []
    # This loop processes each file found.
    for file_path in raw_data_files:
        try:
            print(f"Loading and sampling {os.path.basename(file_path)}...")
            # on_bad_lines='skip' is robust to malformed CSV rows
            df = pd.read_csv(file_path, on_bad_lines='skip')
            
            if 'label' not in df.columns or 'url' not in df.columns:
                print(f"  Warning: Skipping {file_path}. Must contain 'label' and 'url' columns.")
                continue
            
            if len(df) == 0:
                print(f"  Warning: {file_path} is empty. Skipping.")
                continue
                
            # Sample the dataframe using the fraction from config.py
            sample_df = df.sample(frac=fraction, random_state=random_state)
            
            # This check ensures that even if a file is tiny, we get at least one row
            if len(sample_df) == 0 and len(df) > 0:
                print(f"  Warning: Sampling fraction ({fraction}) of {len(df)} rows resulted in 0 samples.")
                print(f"  This happens if (rows * fraction) < 1. Using 1 row as a minimum sample.")
                sample_df = df.sample(n=1, random_state=random_state)
            
            all_samples.append(sample_df)
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    if not all_samples:
        print("Error: No valid data could be loaded.")
        return pd.DataFrame()
        
    # Combine all samples into one big DataFrame
    combined_df = pd.concat(all_samples, ignore_index=True)
    # Shuffle the combined data
    combined_df = combined_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    print(f"Total raw training data prepared: {len(combined_df)} samples.")
    return combined_df

def main():
    print("--- Starting Data Pipeline ---")
    
    # 1. Load and sample raw data
    raw_df = load_and_sample_raw_data(
        data_dir=config.DATA_DIR,
        fraction=config.TRAIN_SAMPLE_FRACTION
    )
    
    if raw_df.empty or len(raw_df) == 0:
        print("Total raw training data is 0 samples. Data pipeline failed. Exiting.")
        return

    # 2. Run full feature extraction
    # This is the slow step
    engineered_df = extract_features_from_dataframe(raw_df)
    
    # 3. Save the final engineered dataset
    engineered_df.to_csv(config.ENGINEERED_TRAIN_FILE, index=False)
    
    print(f"\n--- Data Pipeline Complete ---")
    print(f"Engineered training set saved to: {config.ENGINEERED_TRAIN_FILE}")
    print(f"Total features: {len(config.ALL_FEATURE_COLUMNS)}")

if __name__ == "__main__":
    # Ensure the data directory exists.
    os.makedirs(config.DATA_DIR, exist_ok=True)
    
    # Call the main function.
    # All dummy file creation logic has been removed.
    main()

