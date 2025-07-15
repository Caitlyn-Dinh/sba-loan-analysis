import pandas as pd
import numpy as np
import os

def process_and_combine_csvs(csv_file_paths, target_columns_info, column_mapping=None):
    """
    Reads multiple CSV files, standardizes their structure, and combines them into a single DataFrame.

    Args:
        csv_file_paths (list): A list of paths to the CSV files.
        target_columns_info (dict): A dictionary defining the target column structure.
                                    Keys are column names, values are desired data types (e.g., np.float64, str, 'datetime64[ns]').
        column_mapping (dict, optional): A dictionary to rename columns before standardization.
                                         Keys are original column names, values are new column names.

    Returns:
        pd.DataFrame: A single, combined DataFrame with standardized columns and types.
                      Returns an empty DataFrame if no files are processed or an error occurs.
    """
    combined_df_list = []
    target_column_names = list(target_columns_info.keys())

    print(f"--- Starting CSV Processing and Combination ---")
    print(f"Target Columns and Types: {target_columns_info}")
    if column_mapping:
        print(f"Applying Column Renaming: {column_mapping}")

    for i, file_path in enumerate(csv_file_paths):
        print(f"\nProcessing file {i+1}/{len(csv_file_paths)}: '{file_path}'")
        try:
            # 1. Convert each CSV file into a Pandas DataFrame.
            df = pd.read_csv(file_path)
            print(f"  Original columns: {df.columns.tolist()}")
            print(f"  Original shape: {df.shape}")

            # Apply column renaming if a mapping is provided
            if column_mapping:
                # Only rename columns that actually exist in the current DataFrame
                df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns}, inplace=True)
                print(f"  Columns after renaming: {df.columns.tolist()}")


            # 2. Modify each DataFrame to have the same structure.

            # Identify missing columns in the current DataFrame (after renaming)
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

            # Identify extra columns in the current DataFrame (after renaming)
            extra_cols = [col for col in df.columns if col not in target_column_names]
            if extra_cols:
                print(f"  Dropping extra columns: {extra_cols}")
                df = df.drop(columns=extra_cols)

            # Reorder columns to match the target order
            # Ensure all target columns are present before reordering (even if they were just added as NaN)
            df = df[target_column_names]

            # Convert columns to target data types
            print(f"  Converting column types...")
            for col, dtype in target_columns_info.items():
                if col in df.columns: # Ensure column exists after dropping/adding
                    try:
                        if dtype == 'datetime64[ns]':
                            # Use errors='coerce' to turn unparseable dates into NaT
                            df[col] = pd.to_datetime(df[col], errors='coerce')
                        elif pd.api.types.is_numeric_dtype(dtype) and pd.api.types.is_object_dtype(df[col]):
                            # Attempt to convert object (string) columns to numeric, coercing errors
                            df[col] = pd.to_numeric(df[col], errors='coerce').astype(dtype)
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


# --- Main Execution with User-Provided File Paths ---
if __name__ == "__main__":
    # User-provided CSV file paths for loan data
    csv_files_to_process = [
        "/Users/caitlyndinh/ai-academy/weeks/week4/notebooks/sba_project/2020-present.csv",
        "/Users/caitlyndinh/ai-academy/weeks/week4/notebooks/sba_project/2010-2019.csv",
        "/Users/caitlyndinh/ai-academy/weeks/week4/notebooks/sba_project/2000-2009.csv",
        "/Users/caitlyndinh/ai-academy/weeks/week4/notebooks/sba_project/1991-1999.csv"
    ]

    # Path to your NAICS master file (no longer used for merging, but kept as a variable if needed elsewhere)
    naics_master_file_path = "/Users/caitlyndinh/ai-academy/weeks/week4/notebooks/sba_project/naics_master.csv"

    # Define the mapping from original CSV column names to desired standardized names
    # 'NAICSCode': 'NAICS' mapping has been removed as NAICS is no longer needed.
    column_rename_mapping = {
        'TerminMonths': 'Term',
        'BankName': 'Bank',
        'BorrState': 'State',
        'ChargeoffDate': 'ChgOffDate',
        'GrossChargeoffAmount': 'ChgOffPrinGr',
        'FirstDisbursementDate': 'DisbursementDate',
        'SBAGuaranteedApproval': 'SBA_Appv',
        'LoanStatus': 'MIS_Status',
        # Add other renames if necessary based on your original columns
    }

    # Define the target column structure and data types for SBA loan data
    # 'NAICS' and 'NAICSDescription' have been removed from the target schema.
    target_schema = {
        'GrossApproval': np.float64,
        'ApprovalDate': 'datetime64[ns]',
        'Term': np.int64, # Mapped from TerminMonths
        'State': str, # Mapped from BorrState
        'Bank': str, # Mapped from BankName
        'BankState': str,
        'RevLineCr': str,
        'LowDoc': str,
        'GrAppv': np.float64,
        'SBA_Appv': np.float64, # Mapped from SBAGuaranteedApproval
        'ChgOffDate': 'datetime64[ns]', # Mapped from ChargeoffDate
        'ChgOffPrinGr': np.float64, # Mapped from GrossChargeoffAmount
        'DisbursementDate': 'datetime64[ns]', # Mapped from FirstDisbursementDate
        'DisbursementGross': np.float64,
        'MIS_Status': str, # Mapped from LoanStatus
        # Include other original columns you want to keep and standardize
        'AsofDate': 'datetime64[ns]',
        'Program': str,
        'BorrName': str,
        'BorrStreet': str,
        'BorrCity': str,
        'BorrZip': str,
        'LocationID': str, # Assuming it could be mixed type, keep as string
        'BankFDICNumber': str,
        'BankNCUANumber': str,
        'BankStreet': str,
        'BankCity': str,
        'BankZip': str,
        'ApprovalFY': np.int64,
        'ProcessingMethod': str,
        'Subprogram': str,
        'InitialInterestRate': np.float64,
        'FixedorVariableInterestRate': str,
        'FranchiseCode': str,
        'FranchiseName': str,
        'ProjectCounty': str,
        'ProjectState': str,
        'SBADistrictOffice': str,
        'CongressionalDistrict': str,
        'BusinessType': str,
        'BusinessAge': np.float64, # Assuming numeric, might need conversion from string
        'RevolverStatus': str,
        'JobsSupported': np.float64, # Assuming numeric
        'CollateralInd': str,
        'SoldSecondMarketInd': str
    }


    # Run the processing and combination for loan data
    final_dataframe = process_and_combine_csvs(csv_files_to_process, target_schema, column_mapping=column_rename_mapping)

    if not final_dataframe.empty:
        print("\n--- Initial Combined DataFrame Summary ---")
        print(final_dataframe.head())
        print(final_dataframe.info())
        print(final_dataframe.describe(include='all')) # Use include='all' to see descriptive stats for all columns

        # NAICS master incorporation removed as per user request.
        # The 'incorporate_naics_master' function has also been removed from this code.

    else:
        print("\nNo data was loaded or combined. Please check file paths and permissions.")
