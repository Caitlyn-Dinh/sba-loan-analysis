import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.diagnostic import het_breuschpagan, het_white, linear_reset
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
from sklearn.inspection import permutation_importance # Added for permutation importance

# Set style for plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
# plt.rcParams['font.family'] = 'Inter' # Using Inter font for aesthetics - Commented out to avoid font not found errors

# --- Task 1.1: Data Generation and Exploration ---

# Generate synthetic real estate data
np.random.seed(42)
n_samples = 1000

# Generate features with realistic relationships
# Ensure features are positive and somewhat realistic
house_size = np.random.normal(2000, 600, n_samples)
house_size = np.maximum(500, house_size) # Min size
bedrooms = np.random.poisson(3, n_samples) + 1
bedrooms = np.maximum(1, bedrooms) # Min 1 bedroom
bathrooms = np.random.normal(2.5, 0.8, n_samples)
bathrooms = np.maximum(1, np.round(bathrooms * 2) / 2) # Half-bathrooms, min 1
age = np.random.exponential(20, n_samples)
age = np.maximum(0, age) # Min age 0
distance = np.random.uniform(1, 30, n_samples)
crime_rate = np.random.exponential(5, n_samples)
crime_rate = np.maximum(0.1, crime_rate) # Min crime rate 0.1
school_rating = np.random.uniform(3, 10, n_samples)
tax_rate = np.random.normal(1.5, 0.3, n_samples)
tax_rate = np.maximum(0.5, tax_rate) # Min tax rate 0.5%

# Generate price with realistic relationships and noise
# Base price around $250,000 for average house size, etc.
price = (
    house_size * 150 +             # Larger houses cost more
    bedrooms * 15000 +             # More bedrooms add value
    bathrooms * 12000 +            # More bathrooms add value
    -age * 800 +                   # Older houses slightly less
    -distance * 2000 +             # Further from city, lower price
    -crime_rate * 3000 +           # Higher crime, lower price
    school_rating * 8000 +         # Better schools, higher price
    -tax_rate * 20000 +            # Higher tax, slightly lower net price consideration
    100000 +                       # Base value
    np.random.normal(0, 25000, n_samples) # Add noise
)
price = np.maximum(50000, price) # Ensure price is not negative

# Create DataFrame
real_estate_data = pd.DataFrame({
    'house_size': house_size,
    'bedrooms': bedrooms,
    'bathrooms': bathrooms,
    'age': age,
    'distance_to_city': distance,
    'crime_rate': crime_rate,
    'school_rating': school_rating,
    'tax_rate': tax_rate,
    'price': price
})

print("--- Data Generation Complete ---")
print(real_estate_data.head())
print("\n")

# --- Task 1.1: Exploratory Data Analysis (EDA) ---

print("--- Performing Exploratory Data Analysis ---")

# 1. Distribution plots for all features
print("Generating Distribution Plots for all features...")
for column in real_estate_data.columns:
    plt.figure(figsize=(8, 5))
    sns.histplot(real_estate_data[column], kde=True, bins=30, color='skyblue', edgecolor='black')
    plt.title(f'Distribution of {column.replace("_", " ").title()}', fontsize=14, weight='bold')
    plt.xlabel(column.replace("_", " ").title())
    plt.ylabel('Frequency')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

# 2. Correlation heatmap
print("\nGenerating Correlation Heatmap...")
plt.figure(figsize=(10, 8))
sns.heatmap(real_estate_data.corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, linecolor='black')
plt.title('Correlation Heatmap of Real Estate Data', fontsize=16, weight='bold')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# 3. Scatter plots of price vs each feature
print("\nGenerating Scatter Plots of Price vs. Features...")
features = real_estate_data.columns.drop('price')
for feature in features:
    plt.figure(figsize=(8, 5))
    sns.scatterplot(x=real_estate_data[feature], y=real_estate_data['price'], color='teal', alpha=0.7)
    plt.title(f'Price vs. {feature.replace("_", " ").title()}', fontsize=14, weight='bold')
    plt.xlabel(feature.replace("_", " ").title())
    plt.ylabel('Price')
    plt.grid(axis='both', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

# 4. Identify potential outliers using box plots
print("\nGenerating Box Plots for outlier identification...")
for column in real_estate_data.columns:
    plt.figure(figsize=(8, 5))
    sns.boxplot(y=real_estate_data[column], color='lightcoral')
    plt.title(f'Box Plot of {column.replace("_", " ").title()}', fontsize=14, weight='bold')
    plt.ylabel(column.replace("_", " ").title())
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

print("--- EDA Complete ---")
print("\n")

# --- Task 1.1: Statistical Summary ---

print("--- Performing Statistical Summary ---")

# 1. Descriptive statistics for all variables
print("Descriptive Statistics:\n")
print(real_estate_data.describe().round(2))

# 2. Correlation coefficients with price
print("\nCorrelation Coefficients with Price:\n")
print(real_estate_data.corr()['price'].sort_values(ascending=False).round(3))

# 3. Skewness and kurtosis of the target variable
print(f"\nSkewness of 'price': {real_estate_data['price'].skew():.3f}")
print(f"Kurtosis of 'price': {real_estate_data['price'].kurtosis():.3f}")

print("--- Statistical Summary Complete ---")
print("\n")

# --- Task 1.1: Data Quality Assessment ---

print("--- Performing Data Quality Assessment ---")

# 1. Missing values
print("Missing values per column:\n")
print(real_estate_data.isnull().sum())
print(f"\nTotal missing values: {real_estate_data.isnull().sum().sum()}")
if real_estate_data.isnull().sum().sum() == 0:
    print("No missing values found.")

# 2. Outliers using IQR method
print("\nIdentifying Outliers using IQR method:")
outlier_summary = {}
for column in real_estate_data.columns:
    Q1 = real_estate_data[column].quantile(0.25)
    Q3 = real_estate_data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = real_estate_data[(real_estate_data[column] < lower_bound) | (real_estate_data[column] > upper_bound)]
    if not outliers.empty:
        outlier_summary[column] = len(outliers)
        print(f"  Column '{column}': {len(outliers)} outliers detected.")
    else:
        print(f"  Column '{column}': No outliers detected.")

if not outlier_summary:
    print("No outliers detected across all columns using IQR method.")

# 3. Multicollinearity using correlation analysis (initial check)
print("\nInitial Multicollinearity Check (from Correlation Heatmap):")
high_corr_threshold = 0.7 # Define a threshold for high correlation
corr_matrix = real_estate_data.corr()
for i in range(len(corr_matrix.columns)):
    for j in range(i):
        if abs(corr_matrix.iloc[i, j]) > high_corr_threshold and corr_matrix.columns[i] != 'price' and corr_matrix.columns[j] != 'price':
            print(f"  High correlation between '{corr_matrix.columns[i]}' and '{corr_matrix.columns[j]}': {corr_matrix.iloc[i, j]:.2f}")
if not any(abs(corr_matrix.iloc[i, j]) > high_corr_threshold and corr_matrix.columns[i] != 'price' and corr_matrix.columns[j] != 'price' for i in range(len(corr_matrix.columns)) for j in range(i)):
    print("  No strong multicollinearity observed based on initial correlation analysis (threshold > 0.7).")

print("--- Data Quality Assessment Complete ---")
print("\n")


# --- Task 1.2: Linear Regression Implementation and Validation ---

class LinearRegressionAnalyzer:
    def __init__(self):
        self.model = LinearRegression()
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.X_scaled = None
        self.y_true = None
        self.y_pred = None
        self.residuals = None
        self.sm_model = None # To store statsmodels OLS result for diagnostics
        # Store X_train, y_train for permutation importance
        self.X_train_original = None
        self.y_train_original = None
        
    def fit(self, X, y):
        """Fit the linear regression model with scaling."""
        print("Fitting Linear Regression Model...")
        # Store original training data for permutation importance if needed unscaled
        self.X_train_original = X.copy()
        self.y_train_original = y.copy()

        # Scale the features
        self.X_scaled = self.scaler.fit_transform(X)
        self.y_true = y # Keep y_true as original for residual calculations
        
        # Fit sklearn model
        self.model.fit(self.X_scaled, self.y_true)
        self.is_fitted = True
        self.y_pred = self.model.predict(self.X_scaled)
        self.residuals = self.y_true - self.y_pred

        # Fit statsmodels OLS model for assumption checking and diagnostics
        # Add a constant for the intercept to the scaled X for statsmodels
        X_const = add_constant(self.X_scaled)
        self.sm_model = sm.OLS(self.y_true, X_const).fit()
        print("Model fitted successfully.")
        
    def predict(self, X):
        """Make predictions on new data."""
        if not self.is_fitted:
            raise Exception("Model is not fitted yet. Call fit() first.")
        
        # Scale new X data using the fitted scaler
        X_new_scaled = self.scaler.transform(X)
        return self.model.predict(X_new_scaled)
        
    def check_assumptions(self):
        """Check linear regression assumptions."""
        if not self.is_fitted:
            raise Exception("Model is not fitted yet. Call fit() first.")
        
        print("\n--- Checking Linear Regression Assumptions ---")
        
        # 1. Linearity (scatter plots of residuals vs fitted) - Visual check
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=self.y_pred, y=self.residuals, color='blue', alpha=0.7, ec='black', linewidth=0.5)
        plt.axhline(y=0, color='red', linestyle='--', linewidth=2)
        plt.title('Residuals vs Fitted Values (Linearity Check)', fontsize=14, weight='bold')
        plt.xlabel('Fitted Values')
        plt.ylabel('Residuals')
        plt.grid(axis='both', linestyle='--', alpha=0.7)
        plt.show()
        print("  - Linearity: Visually inspect the 'Residuals vs Fitted Values' plot. A random scatter around zero indicates linearity. Any discernible pattern (e.g., U-shape, funnel shape) suggests non-linearity.")

        # 2. Independence (Durbin-Watson test)
        # The Durbin-Watson statistic is a test for autocorrelation in the residuals.
        # A value close to 2 suggests no autocorrelation. Values less than 1.5 or greater than 2.5 are problematic.
        dw_stat = durbin_watson(self.residuals)
        print(f"  - Independence (Durbin-Watson test): {dw_stat:.3f}")
        if 1.5 < dw_stat < 2.5:
            print("    Result: Residuals appear to be independent (close to 2).")
        else:
            print("    Result: Autocorrelation might be present (value significantly deviates from 2).")

        # 3. Homoscedasticity (Breusch-Pagan test)
        # Null hypothesis: Homoscedasticity (residuals have constant variance).
        # A low p-value (e.g., < 0.05) suggests heteroscedasticity.
        # Need original X with constant added for sm.OLS.
        X_const = add_constant(self.X_scaled)
        lm, lm_pvalue, fvalue, f_pvalue = het_breuschpagan(self.residuals, X_const)
        print(f"  - Homoscedasticity (Breusch-Pagan test): p-value = {lm_pvalue:.3f}")
        if lm_pvalue < 0.05:
            print("    Result: Heteroscedasticity detected (p-value < 0.05).")
        else:
            print("    Result: Homoscedasticity assumed (p-value >= 0.05).")

        # 4. Normality of residuals (Shapiro-Wilk test)
        # Null hypothesis: Residuals are normally distributed.
        # A low p-value (e.g., < 0.05) suggests non-normality.
        shapiro_test = stats.shapiro(self.residuals)
        print(f"  - Normality of Residuals (Shapiro-Wilk test): p-value = {shapiro_test.pvalue:.3f}")
        if shapiro_test.pvalue < 0.05:
            print("    Result: Residuals are not normally distributed (p-value < 0.05).")
        else:
            print("    Result: Residuals appear normally distributed (p-value >= 0.05).")

        print("--- Assumption Checks Complete ---")
        
    def residual_analysis(self):
        """Perform comprehensive residual analysis."""
        if not self.is_fitted:
            raise Exception("Model is not fitted yet. Call fit() first.")
        
        print("\n--- Performing Residual Analysis ---")
        
        # 1. Residuals vs Fitted values (already done in check_assumptions, but good to reiterate)
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=self.y_pred, y=self.residuals, color='blue', alpha=0.7, ec='black', linewidth=0.5)
        plt.axhline(y=0, color='red', linestyle='--', linewidth=2)
        plt.title('Residuals vs Fitted Values', fontsize=14, weight='bold')
        plt.xlabel('Fitted Values')
        plt.ylabel('Residuals')
        plt.grid(axis='both', linestyle='--', alpha=0.7)
        plt.show()

        # 2. Q-Q plot of residuals
        # Compares the distribution of residuals to a normal distribution.
        plt.figure(figsize=(8, 6))
        stats.probplot(self.residuals, dist="norm", plot=plt)
        plt.title('Normal Q-Q Plot of Residuals', fontsize=14, weight='bold')
        plt.grid(axis='both', linestyle='--', alpha=0.7)
        plt.show()
        
        # 3. Scale-Location plot (Spread-Location plot)
        # Plots the square root of the absolute residuals vs fitted values.
        # Helps check homoscedasticity. A horizontal line suggests constant variance.
        sqrt_abs_residuals = np.sqrt(np.abs(self.residuals))
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=self.y_pred, y=sqrt_abs_residuals, color='green', alpha=0.7, ec='black', linewidth=0.5)
        plt.title('Scale-Location Plot (Homoscedasticity Check)', fontsize=14, weight='bold')
        plt.xlabel('Fitted Values')
        plt.ylabel('Square Root of Absolute Residuals')
        plt.grid(axis='both', linestyle='--', alpha=0.7)
        plt.show()

        # 4. Residuals vs Leverage
        # Helps identify influential points.
        # Requires statsmodels OLS results.
        if self.sm_model:
            influence = self.sm_model.get_influence()
            leverage = influence.hat_matrix_diag
            plt.figure(figsize=(10, 6))
            sns.scatterplot(x=leverage, y=self.residuals, color='purple', alpha=0.7, ec='black', linewidth=0.5)
            plt.title('Residuals vs Leverage', fontsize=14, weight='bold')
            plt.xlabel('Leverage')
            plt.ylabel('Residuals')
            plt.grid(axis='both', linestyle='--', alpha=0.7)
            plt.show()
        else:
            print("  Note: statsmodels OLS model not available for Residuals vs Leverage plot.")

        # Additional: Distribution of Residuals
        plt.figure(figsize=(8, 5))
        sns.histplot(self.residuals, kde=True, bins=30, color='lightgreen', edgecolor='black')
        plt.title('Distribution of Residuals', fontsize=14, weight='bold')
        plt.xlabel('Residual Value')
        plt.ylabel('Frequency')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()
        
        print("--- Residual Analysis Complete ---")
        
    def feature_importance_analysis(self, feature_names):
        """Analyze feature importance using coefficients and permutation importance."""
        if not self.is_fitted:
            raise Exception("Model is not fitted yet. Call fit() first.")
            
        print("\n--- Feature Importance Analysis (Coefficients & Permutation) ---")

        # Coefficients from sklearn LinearRegression
        coefficients = self.model.coef_
        intercept = self.model.intercept_
        
        # Create a DataFrame for better readability
        coef_df = pd.DataFrame({'Feature': feature_names, 'Coefficient': coefficients})
        coef_df = coef_df.sort_values(by='Coefficient', ascending=False)
        
        print("Model Coefficients (based on scaled features):")
        print(coef_df.round(4))
        print(f"\nIntercept: {intercept:.4f}")

        print("\nInterpretation of Coefficients:")
        print("  - The magnitude of the coefficient indicates the strength of the relationship with the target variable (price).")
        print("  - The sign (+/-) indicates the direction of the relationship.")
        print("  - A positive coefficient means that as the feature increases, the price tends to increase (e.g., 'house_size').")
        print("  - A negative coefficient means that as the feature increases, the price tends to decrease (e.g., 'distance_to_city', 'crime_rate').")
        print("  - Note: Coefficients are based on scaled features, so their absolute magnitudes are comparable in terms of impact.")
        
        # Plotting coefficients
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Coefficient', y='Feature', data=coef_df, palette='viridis')
        plt.title('Feature Coefficients in Linear Regression', fontsize=16, weight='bold')
        plt.xlabel('Coefficient Value')
        plt.ylabel('Feature')
        plt.tight_layout()
        plt.show()

        # Permutation Importance
        print("\n🔍 Computing Permutation Importance...")
        # Use the original (unscaled) X_train and y_train for permutation importance
        # The model itself handles scaling internally via the analyzer.predict method
        # Or, pass the scaled data and the model that operates on scaled data.
        # It's generally recommended to compute permutation importance on a held-out test set
        # but for simplicity and illustrative purposes within the training context,
        # we'll use a portion of the training data or the full training data.
        # For a more robust measure, use X_test and y_test after fitting.
        
        # Using a subset of training data for faster calculation, or use full data if performance permits
        # Note: For production, use a separate validation/test set for permutation importance.
        pi_results = permutation_importance(
            self.model, self.X_scaled, self.y_true, 
            n_repeats=10, random_state=42, scoring='neg_mean_squared_error' # Using regression scoring
        )
        
        sorted_idx = pi_results.importances_mean.argsort()
        
        plt.figure(figsize=(10, 7))
        plt.boxplot(pi_results.importances[sorted_idx].T,
                    vert=False, labels=np.array(feature_names)[sorted_idx])
        plt.title("Permutation Importance (Training Data)", fontsize=16, weight='bold')
        plt.xlabel("Importance (decrease in MSE)")
        plt.ylabel("Feature")
        plt.tight_layout()
        plt.show()

        # Print top features from permutation importance
        perm_importance_df = pd.DataFrame({
            'Feature': np.array(feature_names)[sorted_idx],
            'Importance_Mean': pi_results.importances_mean[sorted_idx],
            'Importance_Std': pi_results.importances_std[sorted_idx]
        }).round(4)
        print("\nPermutation Importance Summary:")
        print(perm_importance_df)
        print("\nInterpretation of Permutation Importance:")
        print("  - Measures how much the model's performance decreases when a single feature's values are randomly shuffled.")
        print("  - A larger decrease indicates a more important feature.")
        
        print("--- Feature Importance Analysis Complete ---")
        
    def generate_report(self, X_test, y_test):
        """Generate comprehensive model performance report."""
        if not self.is_fitted:
            raise Exception("Model is not fitted yet. Call fit() first.")
            
        print("\n--- Comprehensive Model Performance Report ---")
        
        # Make predictions on the test set
        y_test_pred = self.predict(X_test)
        
        # Performance Metrics
        mse = mean_squared_error(y_test, y_test_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_test_pred)
        r2 = r2_score(y_test, y_test_pred)
        
        print(f"Metrics on Test Set (N={len(y_test)}):")
        print(f"  Mean Squared Error (MSE): {mse:.2f}")
        print(f"  Root Mean Squared Error (RMSE): {rmse:.2f}")
        print(f"  Mean Absolute Error (MAE): {mae:.2f}")
        print(f"  R-squared (R²): {r2:.3f}")

        # Summary from statsmodels OLS (more detailed statistical summary)
        if self.sm_model:
            print("\nDetailed Model Summary (from Statsmodels OLS):")
            print(self.sm_model.summary())
        else:
            print("\nNote: Statsmodels OLS summary is not available. Please ensure it's fitted.")

        # Actual vs Predicted Plot
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=y_test, y=y_test_pred, alpha=0.7, color='orange', ec='black', linewidth=0.5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Ideal Fit')
        plt.title('Actual vs Predicted Prices (Test Set)', fontsize=16, weight='bold')
        plt.xlabel('Actual Price')
        plt.ylabel('Predicted Price')
        plt.grid(axis='both', linestyle='--', alpha=0.7)
        plt.legend()
        plt.tight_layout()
        plt.show()

        print("--- Model Performance Report Complete ---")


# --- Task 1.3: Advanced Diagnostics Functions ---

def detect_influential_points(sm_model, X_df):
    """Detect influential points using Cook's distance and leverage."""
    if not isinstance(sm_model, sm.regression.linear_model.RegressionResultsWrapper):
        raise ValueError("sm_model must be a fitted statsmodels OLS result object.")
        
    print("\n--- Detecting Influential Points (Cook's Distance & Leverage) ---")
    
    influence = sm_model.get_influence()
    
    # Cook's Distance: Measures how much the regression coefficients change if a specific observation is removed.
    # A common rule of thumb for "highly influential" is Cook's D > 4/(n-k-1) or D > 1.
    cooks_d = influence.cooks_distance[0]
    
    # Leverage (Hat values): Measures how far an observation's independent variable values are from the mean.
    leverage = influence.hat_matrix_diag
    
    # Critical Cook's Distance threshold (common: D > 4/(n-k-1) for large N or D > 1)
    n = len(cooks_d)
    k = sm_model.df_model # Number of parameters (features + intercept - 1 if no intercept in X)
    cooks_threshold_large_n = 4 / (n - k - 1) if (n - k - 1) > 0 else 1 # Avoid division by zero
    
    influential_points_cooks = X_df.index[cooks_d > cooks_threshold_large_n].tolist()
    
    # Plot Cook's Distance
    plt.figure(figsize=(10, 6))
    plt.stem(np.arange(len(cooks_d)), cooks_d, markerfmt=",", linefmt="C0-", basefmt=" ")
    plt.axhline(y=cooks_threshold_large_n, color='red', linestyle='--', label=f"Cook's D Threshold ({cooks_threshold_large_n:.3f})")
    plt.title("Cook's Distance", fontsize=14, weight='bold')
    plt.xlabel("Observation Index")
    plt.ylabel("Cook's Distance")
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

    # Plot Leverage
    plt.figure(figsize=(10, 6))
    plt.stem(np.arange(len(leverage)), leverage, markerfmt=",", linefmt="C0-", basefmt=" ")
    plt.axhline(y=2 * (k + 1) / n, color='red', linestyle='--', label=f"Leverage Threshold (2*(k+1)/n)")
    plt.title("Leverage (Hat Values)", fontsize=14, weight='bold')
    plt.xlabel("Observation Index")
    plt.ylabel("Leverage")
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()
    
    print(f"  Number of observations with Cook's Distance > {cooks_threshold_large_n:.3f}: {len(influential_points_cooks)}")
    if influential_points_cooks:
        print(f"  Influential points (indices): {influential_points_cooks}")
    else:
        print("  No highly influential points detected by Cook's Distance threshold.")

    print("--- Influential Point Detection Complete ---")


def multicollinearity_analysis(X_scaled_df):
    """Analyze multicollinearity using VIF (Variance Inflation Factor)."""
    print("\n--- Multicollinearity Analysis (VIF) ---")
    
    # Add constant for intercept term for VIF calculation
    X_const = add_constant(X_scaled_df)
    
    vif_data = pd.DataFrame()
    vif_data["feature"] = X_const.columns
    vif_data["VIF"] = [variance_inflation_factor(X_const.values, i) 
                       for i in range(X_const.shape[1])]
    
    # Drop the constant row, as VIF for constant is not meaningful in this context
    vif_data = vif_data[vif_data['feature'] != 'const']
    vif_data = vif_data.sort_values(by='VIF', ascending=False).round(2)
    
    print("Variance Inflation Factor (VIF) for each feature:\n")
    print(vif_data)
    
    print("\nInterpretation:")
    print("  - VIF values quantify how much the variance of an estimated regression coefficient is inflated due to multicollinearity.")
    print("  - Rule of thumb:")
    print("    - VIF = 1: No multicollinearity.")
    print("    - VIF < 5: Moderate multicollinearity (generally acceptable).")
    print("    - VIF >= 5-10: High multicollinearity (problematic, consider addressing).")
    
    high_vif_features = vif_data[vif_data['VIF'] >= 5]
    if not high_vif_features.empty:
        print("\n  Features with high multicollinearity (VIF >= 5):")
        print(high_vif_features)
    else:
        print("\n  No significant multicollinearity (VIF < 5) detected.")
    
    # Plotting VIFs
    plt.figure(figsize=(10, 6))
    sns.barplot(x='VIF', y='feature', data=vif_data.sort_values(by='VIF', ascending=True), palette='plasma')
    plt.axvline(x=5, color='red', linestyle='--', label='VIF Threshold (5)')
    plt.title('Variance Inflation Factor (VIF) for Features', fontsize=16, weight='bold')
    plt.xlabel('VIF Value')
    plt.ylabel('Feature')
    plt.legend()
    plt.tight_layout()
    plt.show()

    print("--- Multicollinearity Analysis Complete ---")


def heteroscedasticity_tests(sm_model):
    """Perform statistical tests for heteroscedasticity."""
    if not isinstance(sm_model, sm.regression.linear_model.RegressionResultsWrapper):
        raise ValueError("sm_model must be a fitted statsmodels OLS result object.")

    print("\n--- Heteroscedasticity Tests ---")
    
    # Breusch-Pagan Test
    # H0: Homoscedasticity (residuals have constant variance)
    # Ha: Heteroscedasticity
    # A low p-value (e.g., < 0.05) suggests rejecting H0.
    lm, lm_pvalue, fvalue, f_pvalue = het_breuschpagan(sm_model.resid, sm_model.model.exog)
    print(f"  Breusch-Pagan Test:")
    print(f"    Lagrange Multiplier Statistic: {lm:.3f} (p-value: {lm_pvalue:.3f})")
    print(f"    F-Statistic: {fvalue:.3f} (p-value: {f_pvalue:.3f})")
    if lm_pvalue < 0.05:
        print("    Result: Evidence of heteroscedasticity (p < 0.05).")
    else:
        print("    Result: No evidence of heteroscedasticity (p >= 0.05).")
    
    # White Test
    # H0: Homoscedasticity
    # Ha: Heteroscedasticity (more general than Breusch-Pagan)
    # A low p-value (e.g., < 0.05) suggests rejecting H0.
    lm, lm_pvalue, fvalue, f_pvalue = het_white(sm_model.resid, sm_model.model.exog)
    print(f"\n  White Test:")
    print(f"    Lagrange Multiplier Statistic: {lm:.3f} (p-value: {lm_pvalue:.3f})")
    print(f"    F-Statistic: {fvalue:.3f} (p-value: {f_pvalue:.3f})")
    if lm_pvalue < 0.05:
        print("    Result: Evidence of heteroscedasticity (p < 0.05).")
    else:
        print("    Result: No evidence of heteroscedasticity (p >= 0.05).")

    print("--- Heteroscedasticity Tests Complete ---")


def model_specification_tests(sm_model):
    """Test model specification using Ramsey RESET test."""
    if not isinstance(sm_model, sm.regression.linear_model.RegressionResultsWrapper):
        raise ValueError("sm_model must be a fitted statsmodels OLS result object.")
        
    print("\n--- Model Specification Test (Ramsey RESET Test) ---")
    
    # Ramsey RESET test for functional form
    # Null Hypothesis (H0): The model has the correct functional form.
    # Alternative Hypothesis (Ha): The model has an incorrect functional form (e.g., missing higher-order terms).
    # A low p-value (e.g., < 0.05) suggests rejecting H0, indicating specification error.
    reset_test = linear_reset(sm_model, power=[2, 3]) # Use powers 2 and 3 for the test
    
    print(f"  Ramsey RESET Test (p-value): {reset_test.pvalue:.3f}")
    if reset_test.pvalue < 0.05:
        print("    Result: Evidence of model specification error (p < 0.05).")
        print("            Consider adding non-linear terms or interactions.")
    else:
        print("    Result: No evidence of model specification error (p >= 0.05).")

    print("--- Model Specification Test Complete ---")


# --- Example Usage: Integrating all tasks ---

if __name__ == '__main__':
    print("\n=== Starting Linear Regression Pipeline Execution ===")

    # Split data into features (X) and target (y)
    X = real_estate_data.drop('price', axis=1)
    y = real_estate_data['price']
    feature_names = X.columns.tolist()

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"\nTraining set shape: {X_train.shape}, Test set shape: {X_test.shape}")

    # Initialize and fit the Linear Regression Analyzer
    analyzer = LinearRegressionAnalyzer()
    analyzer.fit(X_train, y_train)

    # --- Task 1.2: Validation and Assumption Checks ---
    analyzer.check_assumptions()
    analyzer.residual_analysis()
    analyzer.feature_importance_analysis(feature_names)
    analyzer.generate_report(X_test, y_test)

    # --- Task 1.3: Advanced Diagnostics ---
    # Note: These advanced diagnostics primarily rely on the statsmodels OLS results.
    print("\n=== Running Advanced Diagnostics ===")

    # Create a scaled DataFrame of X_train for VIF and influential points
    # Ensure column names are retained after scaling for VIF.
    X_train_scaled_df = pd.DataFrame(analyzer.X_scaled, columns=feature_names)
    
    detect_influential_points(analyzer.sm_model, X_train_scaled_df)
    multicollinearity_analysis(X_train_scaled_df)
    heteroscedasticity_tests(analyzer.sm_model)
    model_specification_tests(analyzer.sm_model)

    print("\n=== Linear Regression Pipeline Execution Complete ===")
