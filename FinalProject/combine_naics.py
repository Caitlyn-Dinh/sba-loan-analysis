import pandas as pd
import numpy as np
import os

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

def incorporate_naics_master(loan_df, naics_master_file_path):
    """
    Incorporates NAICS industry titles and sector names into the loan DataFrame.

    Args:
        loan_df (pd.DataFrame): The DataFrame containing loan data, including a 'NAICS' column.
        naics_master_file_path (str): Path to the NAICS master CSV file.

    Returns:
        pd.DataFrame: The loan DataFrame enriched with NAICS industry information.
    """
    print(f"\n--- Incorporating NAICS Master File: '{naics_master_file_path}' ---")
    try:
        naics_master_df = pd.read_csv(naics_master_file_path)
        print(f"  NAICS Master file loaded. Columns: {naics_master_df.columns.tolist()}")

        # --- IMPORTANT: Adjusted column names to match your actual NAICS master file ---
        # Based on your output, the columns are 'NAICSCODE', 'NAICSTITLE', 'NAICSVERSION'
        
        # Ensure the NAICS code column in the master file is the same type as in loan_df
        # Assuming 'NAICS' in loan_df is int64, convert master's code column to int64
        if 'NAICSCODE' in naics_master_df.columns: # Changed from 'NAICS_Code' to 'NAICSCODE'
            naics_master_df['NAICSCODE'] = naics_master_df['NAICSCODE'].astype(np.int64)
        else:
            print("  Warning: 'NAICSCODE' column not found in NAICS master file. Please adjust column name in code.")
            return loan_df # Return original if key column is missing

        # Select only the columns needed from the NAICS master for merging
        # Adjusted to use 'NAICSTITLE' and 'NAICSVERSION'
        cols_to_merge = ['NAICSCODE', 'NAICSTITLE', 'NAICSVERSION']
        # Filter to only include columns that actually exist in naics_master_df
        cols_to_merge_existing = [col for col in cols_to_merge if col in naics_master_df.columns]
        
        if not cols_to_merge_existing:
            print("  Warning: No valid columns found in NAICS master file for merging. Please check column names.")
            return loan_df


        # Perform the left merge
        merged_df = pd.merge(
            loan_df,
            naics_master_df[cols_to_merge_existing],
            left_on='NAICS',
            right_on='NAICSCODE', # Changed from 'NAICS_Code' to 'NAICSCODE'
            how='left'
        )

        # Drop the duplicate NAICSCODE column from the master file
        if 'NAICSCODE' in merged_df.columns: # Changed from 'NAICS_Code' to 'NAICSCODE'
            merged_df = merged_df.drop(columns=['NAICSCODE'])

        # Fill NaN values for newly merged columns
        if 'NAICSTITLE' in merged_df.columns: # Changed from 'Industry_Title' to 'NAICSTITLE'
            merged_df['NAICSTITLE'] = merged_df['NAICSTITLE'].fillna('Unknown Industry')
        if 'NAICSVERSION' in merged_df.columns: # Changed from 'Sector_Name' to 'NAICSVERSION'
            merged_df['NAICSVERSION'] = merged_df['NAICSVERSION'].fillna('Unknown NAICS Version') # Updated placeholder

        print(f"  NAICS data incorporated. New shape: {merged_df.shape}")
        print(f"  New columns added: {[col for col in ['NAICSTITLE', 'NAICSVERSION'] if col in merged_df.columns]}") # Updated list
        return merged_df

    except FileNotFoundError:
        print(f"  Error: NAICS Master file not found at '{naics_master_file_path}'. Skipping NAICS incorporation.")
        return loan_df # Return original DataFrame if master file is not found
    except Exception as e:
        print(f"  An error occurred during NAICS master incorporation: {e}. Skipping.")
        return loan_df # Return original DataFrame on error

# --- Main Execution with User-Provided File Paths and NAICS Master Integration ---
if __name__ == "__main__":
    # User-provided CSV file paths for loan data
    csv_files_to_process = [
        "/Users/caitlyndinh/ai-academy/weeks/week4/notebooks/sba_project/2020-present.csv",
        "/Users/caitlyndinh/ai-academy/weeks/week4/notebooks/sba_project/2010-2019.csv",
        "/Users/caitlyndinh/ai-academy/weeks/week4/notebooks/sba_project/2000-2009.csv",
        "/Users/caitlyndinh/ai-academy/weeks/week4/notebooks/sba_project/1991-1999.csv"
    ]

    # Path to your NAICS master file
    naics_master_file_path = "/Users/caitlyndinh/ai-academy/weeks/week4/notebooks/sba_project/naics_master.csv"

    # Define the target column structure and data types for SBA loan data
    target_schema = {
        'LoanNr': np.int64,
        'GrossApproval': np.float64,
        'ApprovalDate': 'datetime64[ns]',
        'Term': np.int64,
        'NAICS': np.int64,
        'State': str,
        'Bank': str,
        'BankState': str,
        'NewExist': np.int64,
        'RevLineCr': str,
        'LowDoc': str,
        'GrAppv': np.float64,
        'SBA_Appv': np.float64,
        'ChgOffDate': 'datetime64[ns]',
        'ChgOffPrinGr': np.float64,
        'Balance': np.float64,
        'DisbursementDate': 'datetime64[ns]',
        'DisbursementGross': np.float64,
        'MIS_Status': str,
        'Default': np.int64
    }

    # Run the processing and combination for loan data
    final_dataframe = process_and_combine_csvs(csv_files_to_process, target_schema)

    if not final_dataframe.empty:
        print("\n--- Initial Combined DataFrame Summary ---")
        print(final_dataframe.head())
        print(final_dataframe.info())
        print(final_dataframe.describe())

        # Incorporate NAICS master data
        final_dataframe = incorporate_naics_master(final_dataframe, naics_master_file_path)

        print("\n--- Final Combined DataFrame with NAICS Info Summary ---")
        print(final_dataframe.head())
        print(final_dataframe.info())
        print(final_dataframe.describe(include='all')) # Use include='all' to see descriptive stats for new string columns too
    else:
        print("\nNo data was loaded or combined. Please check file paths and permissions.")
