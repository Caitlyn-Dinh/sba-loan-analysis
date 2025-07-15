import pandas as pd
import numpy as np
import os # For creating dummy files

def process_and_combine_csvs(csv_file_paths, target_columns_info):
    """
    Reads multiple CSV files, standardizes their structure, and combines them into a single DataFrame.

    Args:
        csv_file_paths (list): A list of paths to the CSV files.
        target_columns_info (dict): A dictionary defining the target column structure.
                                    Keys are column names, values are desired data types (e.g., np.float64, str, 'datetime64[ns]').

    Returns:
        pd.DataFrame: A single, combined DataFrame with standardized columns and types.
                      Returns an empty DataFrame if no files are processed or an error occurs.
    """
    combined_df_list = []
    target_column_names = list(target_columns_info.keys())

    print(f"--- Starting CSV Processing and Combination ---")
    print(f"Target Columns and Types: {target_columns_info}")

    for i, file_path in enumerate(csv_file_paths):
        print(f"\nProcessing file {i+1}/{len(csv_file_paths)}: '{file_path}'")
        try:
            # 1. Convert each CSV file into a Pandas DataFrame.
            df = pd.read_csv(file_path)
            print(f"  Original columns: {df.columns.tolist()}")
            print(f"  Original shape: {df.shape}")

            # 2. Modify each DataFrame to have the same structure.

            # Identify missing columns in the current DataFrame
            missing_cols = [col for col in target_column_names if col not in df.columns]
            if missing_cols:
                print(f"  Adding missing columns: {missing_cols}")
                for col in missing_cols:
                    # Add missing columns with NaN or appropriate default for type
                    if target_columns_info[col] == 'datetime64[ns]':
                        df[col] = pd.NaT # Not a Time for datetime columns
                    elif pd.api.types.is_numeric_dtype(target_columns_info[col]):
                        df[col] = np.nan
                    else: # For object/string types
                        df[col] = None # Or '' depending on preference

            # Identify extra columns in the current DataFrame
            extra_cols = [col for col in df.columns if col not in target_column_names]
            if extra_cols:
                print(f"  Dropping extra columns: {extra_cols}")
                df = df.drop(columns=extra_cols)

            # Reorder columns to match the target order
            df = df[target_column_names]

            # Convert columns to target data types
            print(f"  Converting column types...")
            for col, dtype in target_columns_info.items():
                if col in df.columns: # Ensure column exists after dropping/adding
                    try:
                        if dtype == 'datetime64[ns]':
                            df[col] = pd.to_datetime(df[col], errors='coerce')
                        else:
                            df[col] = df[col].astype(dtype)
                    except Exception as type_err:
                        print(f"    Warning: Could not convert column '{col}' to {dtype} in '{file_path}': {type_err}")
                        # If conversion fails, leave as is or set to NaN/None based on desired error handling
                        if dtype == 'datetime64[ns]':
                            df[col] = pd.NaT
                        elif pd.api.types.is_numeric_dtype(dtype):
                            df[col] = np.nan
                        else:
                            df[col] = None


            combined_df_list.append(df)
            print(f"  Processed shape: {df.shape}")
            print(f"  Processed columns: {df.columns.tolist()}")
            print(f"  Processed dtypes:\n{df.dtypes}")

        except FileNotFoundError:
            print(f"  Error: File not found - '{file_path}'. Skipping.")
        except pd.errors.EmptyDataError:
            print(f"  Warning: Empty CSV file - '{file_path}'. Skipping.")
        except Exception as e:
            print(f"  An unexpected error occurred while processing '{file_path}': {e}. Skipping.")

    # 3. Combine the DataFrames using the pandas.concat() method.
    if combined_df_list:
        final_combined_df = pd.concat(combined_df_list, ignore_index=True)
        print(f"\n--- All files processed. Final Combined DataFrame ---")
        print(f"Final shape: {final_combined_df.shape}")
        print(f"Final columns: {final_combined_df.columns.tolist()}")
        print(f"Final dtypes:\n{final_combined_df.dtypes}")
        return final_combined_df
    else:
        print("\nNo DataFrames were successfully processed and combined.")
        return pd.DataFrame(columns=target_column_names) # Return empty DF with target columns

# --- Example Usage with User-Provided File Paths ---
if __name__ == "__main__":
    # User-provided CSV file paths
    csv_files_to_process = [
        "/Users/caitlyndinh/ai-academy/weeks/week4/notebooks/sba_project/2020-present.csv",
        "/Users/caitlyndinh/ai-academy/weeks/week4/notebooks/sba_project/2010-2019.csv",
        "/Users/caitlyndinh/ai-academy/weeks/week4/notebooks/sba_project/2000-2009.csv",
        "/Users/caitlyndinh/ai-academy/weeks/week4/notebooks/sba_project/1991-1999.csv"
    ]

    # Define the target column structure and data types for SBA loan data
    # This schema is based on common SBA loan dataset columns and types.
    # Adjust these column names and types to precisely match your CSV files.
    target_schema = {
        'LoanNr': np.int64,
        'GrossApproval': np.float64,
        'ApprovalDate': 'datetime64[ns]',
        'Term': np.int64,
        'NAICS': np.int64,
        'State': str,
        'Bank': str,
        'BankState': str,
        'NewExist': np.int64, # New or Existing Business
        'RevLineCr': str, # Revolving Line of Credit (Y/N)
        'LowDoc': str, # LowDoc loan (Y/N)
        'GrAppv': np.float64, # Gross Approved Amount (often same as GrossApproval)
        'SBA_Appv': np.float64, # SBA Approved Amount
        'ChgOffDate': 'datetime64[ns]', # Charge-off Date
        'ChgOffPrinGr': np.float64, # Charge-off Principal Gross
        'Balance': np.float64, # Current Balance (if available)
        'DisbursementDate': 'datetime64[ns]',
        'DisbursementGross': np.float64,
        'MIS_Status': str, # Loan Status (e.g., P I F, CHGOFF)
        'Default': np.int64 # Derived: 1 if defaulted, 0 otherwise
    }

    # Run the processing and combination
    final_dataframe = process_and_combine_csvs(csv_files_to_process, target_schema)

    if not final_dataframe.empty:
        print("\n--- First 5 rows of the Final Combined DataFrame ---")
        print(final_dataframe.head())
        print("\n--- Info of the Final Combined DataFrame ---")
        final_dataframe.info()
        print("\n--- Descriptive Statistics of the Final Combined DataFrame ---")
        print(final_dataframe.describe()) # Added this line
    else:
        print("\nNo data was loaded or combined. Please check file paths and permissions.")

    # No dummy file cleanup needed as real paths are used.
