import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Set style for plots
plt.rcParams['figure.figsize'] = (12, 7)
plt.rcParams['font.family'] = 'sans-serif' # Use a common sans-serif font
plt.style.use('seaborn-v0_8-darkgrid') # Use a clean, modern seaborn style

# --- 1. Data Simulation for SBA 7(a) Loan Pool ---

def simulate_sba_loan_pool(
    num_loans=100,
    avg_loan_amount=500000,
    loan_amount_std=150000,
    avg_interest_rate=0.065, # Annual interest rate
    interest_rate_std=0.005,
    avg_maturity_years=10,
    maturity_std_years=3,
    start_date='2023-01-01',
    cpr_annual=0.08, # Constant Prepayment Rate (annual)
    cdr_annual=0.005, # Constant Default Rate (annual)
    loss_severity=0.40 # Percentage of principal lost on default (e.g., 40%)
):
    """
    Simulates a pool of SBA 7(a) loans with specified characteristics.

    Args:
        num_loans (int): Number of loans in the pool.
        avg_loan_amount (float): Average original loan principal.
        loan_amount_std (float): Standard deviation for loan principal.
        avg_interest_rate (float): Average annual interest rate.
        interest_rate_std (float): Standard deviation for interest rate.
        avg_maturity_years (int): Average loan maturity in years.
        maturity_std_years (int): Standard deviation for maturity in years.
        start_date (str): Start date for loan origination (all loans originate on this date).
        cpr_annual (float): Annual Constant Prepayment Rate (0 to 1).
        cdr_annual (float): Annual Constant Default Rate (0 to 1).
        loss_severity (float): Loss given default (0 to 1).

    Returns:
        pd.DataFrame: DataFrame containing simulated loan pool data.
    """
    print(f"--- Simulating {num_loans} SBA 7(a) Loans ---")
    np.random.seed(42) # For reproducibility

    loan_amounts = np.random.normal(avg_loan_amount, loan_amount_std, num_loans)
    loan_amounts = np.maximum(100000, loan_amounts) # Minimum loan amount

    interest_rates = np.random.normal(avg_interest_rate, interest_rate_std, num_loans)
    interest_rates = np.clip(interest_rates, 0.04, 0.10) # Clip rates to a reasonable range

    maturities_years = np.random.normal(avg_maturity_years, maturity_std_years, num_loans)
    maturities_years = np.maximum(5, np.round(maturities_years)).astype(int) # Min 5 years, round to int

    data = {
        'loan_id': range(1, num_loans + 1),
        'original_balance': loan_amounts,
        'current_balance': loan_amounts, # Initially, current balance is original balance
        'annual_interest_rate': interest_rates,
        'maturity_years': maturities_years,
        'origination_date': pd.to_datetime(start_date)
    }
    df = pd.DataFrame(data)

    df['monthly_interest_rate'] = df['annual_interest_rate'] / 12
    df['maturity_months'] = df['maturity_years'] * 12

    # Calculate initial monthly payment (P & I) for each loan using annuity formula
    # M = P [ i(1 + i)^n ] / [ (1 + i)^n – 1]
    df['scheduled_monthly_payment'] = df.apply(
        lambda row: row['original_balance'] * (row['monthly_interest_rate'] * (1 + row['monthly_interest_rate'])**row['maturity_months']) /
                    ((1 + row['monthly_interest_rate'])**row['maturity_months'] - 1)
                    if row['monthly_interest_rate'] > 0 else row['original_balance'] / row['maturity_months'], # Handle zero interest rate
        axis=1
    )
    # For very short maturities or small balances, this might need adjustment, but for simulation, it's okay.
    # Ensure scheduled payment is not NaN or inf
    df['scheduled_monthly_payment'] = df['scheduled_monthly_payment'].replace([np.inf, -np.inf, np.nan], 0)
    df['scheduled_monthly_payment'] = np.where(df['scheduled_monthly_payment'] < 0, 0, df['scheduled_monthly_payment'])

    print(f"Simulated Pool Total Original Balance: ${df['original_balance'].sum():,.2f}")
    print(f"Simulated Pool Average Loan Amount: ${df['original_balance'].mean():,.2f}")
    print(f"Simulated Pool Average Interest Rate: {df['annual_interest_rate'].mean():.2%}")
    print(f"Simulated Pool Average Maturity: {df['maturity_years'].mean():.1f} years")
    print("--- Simulation Complete ---")
    return df, cpr_annual, cdr_annual, loss_severity

# --- 2. Cash Flow Generation for Loan Pool ---

def project_cash_flows(loan_pool_df, cpr_annual, cdr_annual, projection_months=360):
    """
    Projects monthly cash flows for a pool of loans, considering prepayments and defaults.

    Args:
        loan_pool_df (pd.DataFrame): DataFrame of simulated loan pool.
        cpr_annual (float): Annual Constant Prepayment Rate (0 to 1).
        cdr_annual (float): Annual Constant Default Rate (0 to 1).
        projection_months (int): Number of months to project cash flows.

    Returns:
        pd.DataFrame: DataFrame with monthly cash flow projections (principal, interest, defaults, prepayments).
    """
    print(f"\n--- Projecting Cash Flows for {projection_months} Months ---")

    current_pool = loan_pool_df.copy()
    
    # Monthly rates
    cpr_monthly = 1 - (1 - cpr_annual)**(1/12)
    cdr_monthly = 1 - (1 - cdr_annual)**(1/12)

    monthly_cash_flows = []
    
    # Initialize total outstanding balance
    total_outstanding_balance = current_pool['current_balance'].sum()

    for month in range(1, projection_months + 1):
        if total_outstanding_balance <= 0.01: # Stop if pool is effectively depleted
            print(f"  Pool depleted at month {month-1}. Stopping projection.")
            break

        # Calculate scheduled interest and principal for this month
        interest_paid_this_month = (current_pool['current_balance'] * current_pool['monthly_interest_rate']).sum()
        
        # Scheduled principal is the remainder of scheduled payment after interest
        scheduled_principal_this_month = (current_pool['scheduled_monthly_payment'] - (current_pool['current_balance'] * current_pool['monthly_interest_rate'])).sum()
        scheduled_principal_this_month = np.maximum(0, scheduled_principal_this_month) # Ensure non-negative

        # Calculate defaults
        defaults_principal = (current_pool['current_balance'] * cdr_monthly).sum()
        
        # Calculate prepayments
        prepayments_principal = ((current_pool['current_balance'] - (current_pool['current_balance'] * cdr_monthly)) * cpr_monthly).sum()
        
        # Update current balances for the next month
        current_pool['current_balance'] = current_pool['current_balance'] - \
                                          (current_pool['scheduled_monthly_payment'] - (current_pool['current_balance'] * current_pool['monthly_interest_rate'])) - \
                                          (current_pool['current_balance'] * cdr_monthly) - \
                                          ((current_pool['current_balance'] - (current_pool['current_balance'] * cdr_monthly)) * cpr_monthly)
        
        # Ensure balances don't go negative due to rounding/over-amortization
        current_pool['current_balance'] = np.maximum(0, current_pool['current_balance'])

        # Loans that have reached maturity are considered fully paid
        current_pool.loc[current_pool['maturity_months'] <= month, 'current_balance'] = 0

        # Calculate actual principal received (scheduled + prepayments) - defaults
        actual_principal_received = scheduled_principal_this_month + prepayments_principal - defaults_principal * loss_severity

        monthly_cash_flows.append({
            'month': month,
            'beginning_balance': total_outstanding_balance,
            'interest_cash_flow': interest_paid_this_month,
            'scheduled_principal_cash_flow': scheduled_principal_this_month,
            'prepayment_cash_flow': prepayments_principal,
            'default_principal_amount': defaults_principal, # Principal amount defaulted
            'net_principal_cash_flow': actual_principal_received # Principal after defaults
        })
        
        # Update total outstanding balance for next iteration
        total_outstanding_balance = current_pool['current_balance'].sum()

    cash_flows_df = pd.DataFrame(monthly_cash_flows)
    
    # Calculate total cash flow received by investor
    cash_flows_df['total_investor_cash_flow'] = cash_flows_df['interest_cash_flow'] + \
                                                  cash_flows_df['net_principal_cash_flow']
    
    print("--- Cash Flow Projection Complete ---")
    return cash_flows_df

# --- 3. Valuation of Loan Pool / Securities ---

def value_loan_pool(cash_flows_df, discount_rate_annual):
    """
    Calculates the Present Value (PV) of the projected cash flows.

    Args:
        cash_flows_df (pd.DataFrame): DataFrame with monthly cash flow projections.
        discount_rate_annual (float): Annual discount rate (e.g., market yield for similar assets).

    Returns:
        float: Present Value (PV) of the loan pool.
    """
    print(f"\n--- Valuing Loan Pool with Annual Discount Rate: {discount_rate_annual:.2%} ---")
    
    if cash_flows_df.empty:
        print("No cash flows to value. PV = $0.00")
        return 0.0

    discount_rate_monthly = (1 + discount_rate_annual)**(1/12) - 1
    
    # Calculate discount factors for each month
    cash_flows_df['discount_factor'] = 1 / (1 + discount_rate_monthly)**cash_flows_df['month']
    
    # Calculate discounted cash flows
    cash_flows_df['discounted_cash_flow'] = cash_flows_df['total_investor_cash_flow'] * cash_flows_df['discount_factor']
    
    total_pv = cash_flows_df['discounted_cash_flow'].sum()
    
    print(f"Total Present Value (PV) of the Loan Pool: ${total_pv:,.2f}")
    print("--- Valuation Complete ---")
    return total_pv

# --- 4. Visualization of Cash Flows ---

def plot_cash_flows(cash_flows_df):
    """
    Plots the projected cash flow components over time.
    """
    print("\n--- Generating Cash Flow Visualizations ---")
    plt.figure(figsize=(14, 8))
    
    plt.plot(cash_flows_df['month'], cash_flows_df['interest_cash_flow'], label='Interest', color='green', linewidth=2)
    plt.plot(cash_flows_df['month'], cash_flows_df['scheduled_principal_cash_flow'], label='Scheduled Principal', color='blue', linestyle='--', linewidth=2)
    plt.plot(cash_flows_df['month'], cash_flows_df['prepayment_cash_flow'], label='Prepayments', color='orange', linewidth=2)
    plt.plot(cash_flows_df['month'], -cash_flows_df['default_principal_amount'], label='Defaults (Principal)', color='red', linestyle=':', linewidth=2) # Defaults as negative
    plt.plot(cash_flows_df['month'], cash_flows_df['total_investor_cash_flow'], label='Total Investor Cash Flow', color='purple', linewidth=3, alpha=0.7)
    
    plt.title('Projected Monthly Cash Flows for SBA 7(a) Loan Pool', fontsize=16, weight='bold')
    plt.xlabel('Month', fontsize=12)
    plt.ylabel('Cash Flow ($)', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xlim(0, cash_flows_df['month'].max() * 1.05) # Add a little buffer to x-axis
    plt.tight_layout()
    plt.show()
    print("--- Cash Flow Visualizations Complete ---")


# --- Main Execution ---

if __name__ == "__main__":
    print("=========================================================")
    print("=== SBA 7(a) Loan Pool & Securities Valuation Pipeline ===")
    print("=========================================================\n")

    # Step 1: Simulate the SBA 7(a) Loan Pool
    initial_loan_pool_df, cpr, cdr, loss_severity = simulate_sba_loan_pool(
        num_loans=500,
        avg_loan_amount=250000,
        loan_amount_std=100000,
        avg_interest_rate=0.07,
        interest_rate_std=0.007,
        avg_maturity_years=15,
        maturity_std_years=5,
        cpr_annual=0.10, # Example CPR
        cdr_annual=0.008, # Example CDR
        loss_severity=0.50 # Example Loss Severity
    )

    # Step 2: Project Cash Flows
    # We project for a longer period to capture most of the cash flows, e.g., max maturity + buffer
    max_projection_months = initial_loan_pool_df['maturity_months'].max() + 60 # Max maturity plus 5 years buffer
    
    projected_cash_flows_df = project_cash_flows(
        initial_loan_pool_df,
        cpr,
        cdr,
        projection_months=max_projection_months
    )

    if not projected_cash_flows_df.empty:
        # Step 3: Value the Loan Pool
        discount_rate = 0.05 # Example annual discount rate (e.g., benchmark bond yield)
        pool_present_value = value_loan_pool(projected_cash_flows_df, discount_rate)

        # Step 4: Visualize Cash Flows
        plot_cash_flows(projected_cash_flows_df)
    else:
        print("\nSkipping valuation and visualization due to empty cash flow projection.")

    print("\n=========================================================")
    print("=== Pipeline Execution Complete ===")
    print("=========================================================")
