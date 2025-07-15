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
    loss_severity=0.40, # Percentage of principal lost on default (e.g., 40%)
    guarantee_percentage=0.75, # Percentage of loan guaranteed by SBA
    upfront_guarantee_fee_rate=0.03 # Upfront fee as % of guaranteed portion
):
    """
    Simulates a pool of SBA 7(a) loans with specified characteristics and calculates upfront fees.

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
        guarantee_percentage (float): Percentage of the loan guaranteed by SBA (0 to 1).
        upfront_guarantee_fee_rate (float): Upfront fee rate applied to the guaranteed portion.

    Returns:
        tuple: (pd.DataFrame: Simulated loan pool data,
                float: cpr_annual, float: cdr_annual, float: loss_severity,
                float: guarantee_percentage, float: upfront_guarantee_fee_rate)
    """
    print(f"--- Simulating {num_loans} SBA 7(a) Loans ---")
    np.random.seed(42) # For reproducibility in the base loan pool generation

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
        lambda row: row['original_balance'] * (row['monthly_interest_rate'] * (1 + row['monthly_interest_rate'])**row['maturity_months']) / \
                    ((1 + row['monthly_interest_rate'])**row['maturity_months'] - 1)
                    if row['monthly_interest_rate'] > 0 else row['original_balance'] / row['maturity_months'], # Handle zero interest rate
        axis=1
    )
    df['scheduled_monthly_payment'] = df['scheduled_monthly_payment'].replace([np.inf, -np.inf, np.nan], 0)
    df['scheduled_monthly_payment'] = np.where(df['scheduled_monthly_payment'] < 0, 0, df['scheduled_monthly_payment'])

    # Calculate upfront guarantee fees
    df['guaranteed_portion'] = df['original_balance'] * guarantee_percentage
    df['upfront_guarantee_fee'] = df['guaranteed_portion'] * upfront_guarantee_fee_rate
    total_upfront_fees = df['upfront_guarantee_fee'].sum()

    print(f"Simulated Pool Total Original Balance: ${df['original_balance'].sum():,.2f}")
    print(f"Simulated Pool Average Loan Amount: ${df['original_balance'].mean():,.2f}")
    print(f"Simulated Pool Average Interest Rate: {df['annual_interest_rate'].mean():.2%}")
    print(f"Simulated Pool Average Maturity: {df['maturity_years'].mean():.1f} years")
    print(f"Total Upfront Guarantee Fees: ${total_upfront_fees:,.2f}")
    print("--- Simulation Complete ---")
    return df, cpr_annual, cdr_annual, loss_severity, guarantee_percentage, upfront_guarantee_fee_rate

# --- 2. Cash Flow Generation for Loan Pool ---

def project_cash_flows(loan_pool_df, cpr_annual, cdr_annual, loss_severity, guarantee_percentage, ongoing_servicing_fee_rate_annual, projection_months=360):
    """
    Projects monthly cash flows for a pool of loans, considering prepayments, defaults, and ongoing fees.

    Args:
        loan_pool_df (pd.DataFrame): DataFrame of simulated loan pool (must contain 'upfront_guarantee_fee').
        cpr_annual (float): Annual Constant Prepayment Rate (0 to 1).
        cdr_annual (float): Annual Constant Default Rate (0 to 1).
        loss_severity (float): Loss given default (0 to 1).
        guarantee_percentage (float): Percentage of the loan guaranteed by SBA (0 to 1).
        ongoing_servicing_fee_rate_annual (float): Annual ongoing fee rate on guaranteed outstanding balance.
        projection_months (int): Number of months to project cash flows.

    Returns:
        tuple: (pd.DataFrame: Monthly cash flow projections,
                float: total_projected_losses,
                float: total_projected_fees)
    """
    print(f"\n--- Projecting Cash Flows for {projection_months} Months (CPR: {cpr_annual:.2%}, CDR: {cdr_annual:.2%}) ---")

    current_pool = loan_pool_df.copy()
    
    # Monthly rates
    cpr_monthly = 1 - (1 - cpr_annual)**(1/12)
    cdr_monthly = 1 - (1 - cdr_annual)**(1/12)
    ongoing_servicing_fee_rate_monthly = ongoing_servicing_fee_rate_annual / 12

    monthly_cash_flows = []
    total_losses_incurred = 0.0
    total_ongoing_fees_collected = 0.0

    # Start with upfront fees as initial fee income
    total_fees_collected = loan_pool_df['upfront_guarantee_fee'].sum()
    
    # Initialize total outstanding balance
    total_outstanding_balance = current_pool['current_balance'].sum()

    for month in range(1, projection_months + 1):
        if total_outstanding_balance <= 0.01 and current_pool['current_balance'].sum() <= 0.01: # Stop if pool is effectively depleted
            print(f"  Pool effectively depleted at month {month-1}. Stopping projection.")
            break

        # Calculate scheduled interest and principal for this month
        interest_paid_this_month = (current_pool['current_balance'] * current_pool['monthly_interest_rate']).sum()
        
        # Scheduled principal is the remainder of scheduled payment after interest
        scheduled_principal_this_month = (current_pool['scheduled_monthly_payment'] - (current_pool['current_balance'] * current_pool['monthly_interest_rate'])).sum()
        scheduled_principal_this_month = np.maximum(0, scheduled_principal_this_month) # Ensure non-negative

        # Calculate defaults
        defaults_principal_this_month_per_loan = current_pool['current_balance'] * cdr_monthly
        defaults_principal_this_month = defaults_principal_this_month_per_loan.sum()
        
        # Calculate prepayments (applied to non-defaulted balance)
        prepayments_principal_this_month_per_loan = (current_pool['current_balance'] - defaults_principal_this_month_per_loan) * cpr_monthly
        prepayments_principal_this_month = prepayments_principal_this_month_per_loan.sum()

        # Calculate ongoing servicing fees for this month
        # Fees are typically on the guaranteed portion of the outstanding balance
        ongoing_fees_this_month = (current_pool['current_balance'] * guarantee_percentage * ongoing_servicing_fee_rate_monthly).sum()
        total_ongoing_fees_collected += ongoing_fees_this_month
        total_fees_collected += ongoing_fees_this_month # Add to total fees

        # Update current balances for the next month
        current_pool['current_balance'] = current_pool['current_balance'] - \
                                          (current_pool['scheduled_monthly_payment'] - (current_pool['current_balance'] * current_pool['monthly_interest_rate'])) - \
                                          defaults_principal_this_month_per_loan - \
                                          prepayments_principal_this_month_per_loan
        
        # Ensure balances don't go negative due to rounding/over-amortization
        current_pool['current_balance'] = np.maximum(0, current_pool['current_balance'])

        # Loans that have reached maturity are considered fully paid
        current_pool.loc[current_pool['maturity_months'] <= month, 'current_balance'] = 0

        # Calculate actual principal received by investor (scheduled + prepayments) - actual losses from defaults
        # The loss is (defaults_principal * loss_severity)
        actual_losses_this_month = defaults_principal_this_month * loss_severity
        total_losses_incurred += actual_losses_this_month # Accumulate total losses

        net_principal_cash_flow = scheduled_principal_this_month + prepayments_principal_this_month - actual_losses_this_month

        monthly_cash_flows.append({
            'month': month,
            'beginning_balance': total_outstanding_balance,
            'interest_cash_flow': interest_paid_this_month,
            'scheduled_principal_cash_flow': scheduled_principal_this_month,
            'prepayment_cash_flow': prepayments_principal_this_month,
            'default_principal_amount': defaults_principal_this_month, # Principal amount defaulted
            'actual_loss_incurred': actual_losses_this_month, # Actual dollar loss from defaults
            'net_principal_cash_flow': net_principal_cash_flow, # Principal after losses
            'ongoing_servicing_fees': ongoing_fees_this_month # Fees collected this month
        })
        
        # Update total outstanding balance for next iteration
        total_outstanding_balance = current_pool['current_balance'].sum()

    cash_flows_df = pd.DataFrame(monthly_cash_flows)
    
    # Calculate total cash flow received by investor (excluding fees which are separate income)
    cash_flows_df['total_investor_cash_flow'] = cash_flows_df['interest_cash_flow'] + \
                                                  cash_flows_df['net_principal_cash_flow']
    
    print("--- Cash Flow Projection Complete ---")
    return cash_flows_df, total_losses_incurred, total_fees_collected

# --- 3. Valuation of Loan Pool / Securities ---

def value_loan_pool(cash_flows_df, discount_rate_annual, scenario_name="Scenario"):
    """
    Calculates the Present Value (PV) of the projected cash flows.

    Args:
        cash_flows_df (pd.DataFrame): DataFrame with monthly cash flow projections.
        discount_rate_annual (float): Annual discount rate (e.g., market yield for similar assets).
        scenario_name (str): Name of the scenario for print output.

    Returns:
        float: Present Value (PV) of the loan pool.
    """
    # print(f"\n--- Valuing Loan Pool for {scenario_name} with Annual Discount Rate: {discount_rate_annual:.2%} ---")
    
    if cash_flows_df.empty:
        # print("No cash flows to value. PV = $0.00")
        return 0.0

    discount_rate_monthly = (1 + discount_rate_annual)**(1/12) - 1
    
    # Calculate discount factors for each month
    cash_flows_df['discount_factor'] = 1 / (1 + discount_rate_monthly)**cash_flows_df['month']
    
    # Calculate discounted cash flows
    cash_flows_df['discounted_cash_flow'] = cash_flows_df['total_investor_cash_flow'] * cash_flows_df['discount_factor']
    
    total_pv = cash_flows_df['discounted_cash_flow'].sum()
    
    # print(f"Total Present Value (PV) of the Loan Pool for {scenario_name}: ${total_pv:,.2f}")
    # print("--- Valuation Complete ---")
    return total_pv

# --- 4. Visualization of Cash Flows (Optional for individual scenarios, but useful for debugging) ---

def plot_cash_flows(cash_flows_df, title_suffix=""):
    """
    Plots the projected cash flow components over time.
    """
    print(f"\n--- Generating Cash Flow Visualizations {title_suffix} ---")
    plt.figure(figsize=(14, 8))
    
    plt.plot(cash_flows_df['month'], cash_flows_df['interest_cash_flow'], label='Interest', color='green', linewidth=2)
    plt.plot(cash_flows_df['month'], cash_flows_df['scheduled_principal_cash_flow'], label='Scheduled Principal', color='blue', linestyle='--', linewidth=2)
    plt.plot(cash_flows_df['month'], cash_flows_df['prepayment_cash_flow'], label='Prepayments', color='orange', linewidth=2)
    plt.plot(cash_flows_df['month'], -cash_flows_df['actual_loss_incurred'], label='Actual Losses (Principal)', color='red', linestyle=':', linewidth=2) # Losses as negative
    plt.plot(cash_flows_df['month'], cash_flows_df['ongoing_servicing_fees'], label='Ongoing Servicing Fees', color='cyan', linestyle='-.', linewidth=2)
    plt.plot(cash_flows_df['month'], cash_flows_df['total_investor_cash_flow'], label='Total Investor Cash Flow', color='purple', linewidth=3, alpha=0.7)
    
    plt.title(f'Projected Monthly Cash Flows for SBA 7(a) Loan Pool {title_suffix}', fontsize=16, weight='bold')
    plt.xlabel('Month', fontsize=12)
    plt.ylabel('Cash Flow ($)', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xlim(0, cash_flows_df['month'].max() * 1.05) # Add a little buffer to x-axis
    plt.tight_layout()
    plt.show()
    print("--- Cash Flow Visualizations Complete ---")


# --- New Function for Running a Single Scenario ---

def run_single_scenario(
    base_loan_pool_df, # Use a pre-simulated base pool for consistency across scenarios
    scenario_params,
    projection_months=360
):
    """
    Runs a single scenario of cash flow projection and valuation.

    Args:
        base_loan_pool_df (pd.DataFrame): The initial, fixed loan pool DataFrame.
        scenario_params (dict): Dictionary of parameters for this specific scenario,
                                e.g., {'cpr_annual': 0.10, 'cdr_annual': 0.005,
                                       'loss_severity': 0.40, 'discount_rate': 0.05,
                                       'guarantee_percentage': 0.75,
                                       'upfront_guarantee_fee_rate': 0.03,
                                       'ongoing_servicing_fee_rate_annual': 0.0055}
        projection_months (int): Number of months to project cash flows.

    Returns:
        dict: A dictionary containing results for this scenario:
              'pv', 'total_losses', 'total_fees', 'success_score', and input parameters.
    """
    cpr = scenario_params['cpr_annual']
    cdr = scenario_params['cdr_annual']
    loss_severity = scenario_params['loss_severity']
    discount_rate = scenario_params['discount_rate']
    guarantee_percentage = scenario_params['guarantee_percentage']
    upfront_guarantee_fee_rate = scenario_params['upfront_guarantee_fee_rate']
    ongoing_servicing_fee_rate_annual = scenario_params['ongoing_servicing_fee_rate_annual']

    # Ensure upfront fees are included in the base_loan_pool_df if not already
    # This assumes base_loan_pool_df comes from simulate_sba_loan_pool which now calculates it.
    # If not, you'd need to calculate it here:
    # base_loan_pool_df['guaranteed_portion'] = base_loan_pool_df['original_balance'] * guarantee_percentage
    # base_loan_pool_df['upfront_guarantee_fee'] = base_loan_pool_df['guaranteed_portion'] * upfront_guarantee_fee_rate

    projected_cash_flows_df, total_losses, total_fees = project_cash_flows(
        base_loan_pool_df,
        cpr,
        cdr,
        loss_severity,
        guarantee_percentage,
        ongoing_servicing_fee_rate_annual,
        projection_months=projection_months
    )

    pv = value_loan_pool(projected_cash_flows_df, discount_rate)

    # Calculate success score: Ratio of total fees to total losses.
    # Add a small epsilon to total_losses to avoid division by zero if losses are 0.
    success_score = total_fees / (total_losses + 1e-9) if total_losses > 0 else (1.0 if total_fees > 0 else 0.0)

    return {
        'cpr_annual': cpr,
        'cdr_annual': cdr,
        'loss_severity': loss_severity,
        'discount_rate': discount_rate,
        'guarantee_percentage': guarantee_percentage,
        'upfront_guarantee_fee_rate': upfront_guarantee_fee_rate,
        'ongoing_servicing_fee_rate_annual': ongoing_servicing_fee_rate_annual,
        'present_value': pv,
        'total_projected_losses': total_losses,
        'total_projected_fees': total_fees,
        'success_score': success_score # Ratio of fees to losses
    }


# --- Main Execution for Scenario Analysis ---

if __name__ == "__main__":
    print("=========================================================")
    print("=== SBA 7(a) Loan Pool Actuarial Soundness Analysis ===")
    print("=========================================================\n")

    # --- Step 1: Simulate a Base SBA 7(a) Loan Pool (fixed for all scenarios) ---
    # We simulate the loan pool once to keep the underlying loans consistent
    # across different scenarios, allowing us to isolate the impact of changing
    # economic/behavioral parameters.
    base_loan_pool_params = {
        'num_loans': 500,
        'avg_loan_amount': 250000,
        'loan_amount_std': 100000,
        'avg_interest_rate': 0.07,
        'interest_rate_std': 0.007,
        'avg_maturity_years': 15,
        'maturity_std_years': 5,
        'cpr_annual': 0.10, # Base CPR
        'cdr_annual': 0.008, # Base CDR
        'loss_severity': 0.50, # Base Loss Severity
        'guarantee_percentage': 0.75, # Base Guarantee Percentage
        'upfront_guarantee_fee_rate': 0.03 # Base Upfront Fee Rate
    }
    initial_loan_pool_df, _, _, _, _, _ = simulate_sba_loan_pool(**base_loan_pool_params)

    max_projection_months = initial_loan_pool_df['maturity_months'].max() + 60 # Max maturity plus 5 years buffer

    # --- Step 2: Define Scenario Parameters and Ranges ---
    num_scenarios = 1000
    scenario_results = []

    # Define ranges for parameters to vary
    cpr_range = (0.05, 0.20) # CPR from 5% to 20%
    cdr_range = (0.002, 0.015) # CDR from 0.2% to 1.5%
    discount_rate_range = (0.04, 0.08) # Discount rate from 4% to 8%
    loss_severity_range = (0.30, 0.60) # Loss severity from 30% to 60%
    # We can also vary fee rates if desired, but for this example, let's keep them fixed
    # to focus on the impact of risk parameters.
    fixed_ongoing_servicing_fee_rate_annual = 0.0055 # 0.55% annual

    print(f"\n--- Running {num_scenarios} Scenarios ---")
    for i in range(num_scenarios):
        # Generate random parameters for each scenario
        scenario_cpr = np.random.uniform(*cpr_range)
        scenario_cdr = np.random.uniform(*cdr_range)
        scenario_discount_rate = np.random.uniform(*discount_rate_range)
        scenario_loss_severity = np.random.uniform(*loss_severity_range)

        current_scenario_params = {
            'cpr_annual': scenario_cpr,
            'cdr_annual': scenario_cdr,
            'loss_severity': scenario_loss_severity,
            'discount_rate': scenario_discount_rate,
            'guarantee_percentage': base_loan_pool_params['guarantee_percentage'], # Keep fixed
            'upfront_guarantee_fee_rate': base_loan_pool_params['upfront_guarantee_fee_rate'], # Keep fixed
            'ongoing_servicing_fee_rate_annual': fixed_ongoing_servicing_fee_rate_annual # Keep fixed
        }

        # Run the single scenario
        result = run_single_scenario(
            initial_loan_pool_df,
            current_scenario_params,
            projection_months=max_projection_months
        )
        scenario_results.append(result)

        if (i + 1) % 100 == 0:
            print(f"  Completed {i + 1}/{num_scenarios} scenarios...")

    results_df = pd.DataFrame(scenario_results)
    print(f"--- {num_scenarios} Scenarios Completed ---")

    # --- Step 3: Analyze and Visualize Scenario Results ---
    print("\n--- Analyzing Scenario Results ---")

    # Calculate summary statistics for key metrics
    print(f"Average Present Value (PV) across scenarios: ${results_df['present_value'].mean():,.2f}")
    print(f"Average Success Score (Fees/Losses) across scenarios: {results_df['success_score'].mean():.2f}")
    print(f"Number of 'Successful' Scenarios (Score >= 1): {results_df[results_df['success_score'] >= 1].shape[0]}")
    print(f"Percentage of 'Successful' Scenarios: {results_df[results_df['success_score'] >= 1].shape[0] / num_scenarios:.2%}")

    # Plotting distributions of key outcomes
    plt.figure(figsize=(18, 6))

    # Histogram of Present Value
    plt.subplot(1, 2, 1)
    plt.hist(results_df['present_value'], bins=30, color='skyblue', edgecolor='black')
    plt.title('Distribution of Loan Pool Present Value Across Scenarios', fontsize=14)
    plt.xlabel('Present Value ($)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.ticklabel_format(style='plain', axis='x') # Prevent scientific notation on x-axis

    # Histogram of Success Score (Fees / Losses)
    plt.subplot(1, 2, 2)
    plt.hist(results_df['success_score'], bins=30, color='lightgreen', edgecolor='black')
    plt.axvline(1.0, color='red', linestyle='dashed', linewidth=2, label='Break-even (Fees/Losses = 1)')
    plt.title('Distribution of Actuarial Success Score (Fees / Losses)', fontsize=14)
    plt.xlabel('Success Score (Total Fees / Total Losses)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

    # Optional: Scatter plot to see relationship between a parameter and success score
    plt.figure(figsize=(10, 6))
    plt.scatter(results_df['cpr_annual'], results_df['success_score'], alpha=0.5, color='blue')
    plt.axhline(1.0, color='red', linestyle='dashed', linewidth=2, label='Break-even')
    plt.title('Success Score vs. Annual Prepayment Rate (CPR)', fontsize=14)
    plt.xlabel('Annual Prepayment Rate (CPR)', fontsize=12)
    plt.ylabel('Success Score (Fees / Losses)', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

    print("\n=========================================================")
    print("=== Scenario Analysis Complete ===")
    print("=========================================================")
