import pandas as pd

def display_gross_approval_summary(file_path):
    """
    Loads a CSV file and displays summary statistics for the 'GrossApproval' column.

    Args:
        file_path (str): The path to the CSV data file.
    """
    try:
        # Load the CSV file into a pandas DataFrame
        df = pd.read_csv(file_path)

        # Check if 'GrossApproval' column exists
        if 'GrossApproval' in df.columns:
            print(f"--- Summary Statistics for 'GrossApproval' from '{file_path}' ---")
            # Display summary statistics for the 'GrossApproval' column
            # .describe() provides count, mean, std, min, 25%, 50%, 75%, max
            print(df['GrossApproval'].describe())
            print("\n------------------------------------------------------------------")
        else:
            print(f"Error: 'GrossApproval' column not found in the file '{file_path}'.")
            print(f"Available columns are: {df.columns.tolist()}")

    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found. Please check the path.")
    except Exception as e:
        print(f"An error occurred: {e}")

# --- Example Usage ---
# IMPORTANT: Replace 'your_data_file.csv' with the actual path to your CSV file.
# For example: 'data/sba_loans.csv' or '/Users/yourname/Documents/sba_data.csv'
csv_file_path = '/Users/caitlyndinh/ai-academy/FinalProject/sba_project/2020-present.csv' # <--- CHANGE THIS TO YOUR ACTUAL FILE PATH

display_gross_approval_summary(csv_file_path)

# --- You can also create a dummy CSV for testing if you don't have one ---
# import os
# dummy_data = {
#     'LoanID': [1, 2, 3, 4, 5],
#     'GrossApproval': [150000, 250000, 50000, 300000, 180000],
#     'InterestRate': [0.05, 0.06, 0.07, 0.055, 0.065]
# }
# dummy_df = pd.DataFrame(dummy_data)
# dummy_csv_path = 'dummy_sba_data.csv'
# dummy_df.to_csv(dummy_csv_path, index=False)
# print(f"\n--- Running with dummy data file: {dummy_csv_path} ---")
# display_gross_approval_summary(dummy_csv_path)
# # Clean up dummy file
# # os.remove(dummy_csv_path)
