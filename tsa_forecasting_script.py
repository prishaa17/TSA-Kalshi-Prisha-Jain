import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Constants
RANDOM_STATE = 42
VALIDATION_DAYS = 90
TRAIN_PATH = '/mnt/user-data/uploads/tsa_train.csv'
TEST_PATH = '/mnt/user-data/uploads/tsa_test.csv'
OUTPUT_PATH = '/mnt/user-data/outputs/tsa_test_predictions.csv'

# Set random seed for reproducibility
np.random.seed(RANDOM_STATE)

# Configure plotting
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def load_data(train_path, test_path):
    """
    Load and prepare TSA passenger volume data.
    
    Parameters:
    -----------
    train_path : str
        Path to training CSV file
    test_path : str
        Path to test CSV file
        
    Returns:
    --------
    train_df : pd.DataFrame
        Training data with parsed dates
    test_df : pd.DataFrame
        Test data with parsed dates
    """
    print("="*60)
    print("STEP 1: Loading Data")
    print("="*60)
    
    # Load data
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    # Parse dates
    train_df['Date'] = pd.to_datetime(train_df['Date'])
    test_df['Date'] = pd.to_datetime(test_df['Date'])
    
    # Sort chronologically (critical for time-series)
    train_df = train_df.sort_values('Date').reset_index(drop=True)
    test_df = test_df.sort_values('Date').reset_index(drop=True)
    
    # Print summary
    print(f"\nTraining data shape: {train_df.shape}")
    print(f"  Date range: {train_df['Date'].min()} to {train_df['Date'].max()}")
    print(f"  Missing values: {train_df.isnull().sum().sum()}")
    
    print(f"\nTest data shape: {test_df.shape}")
    print(f"  Date range: {test_df['Date'].min()} to {test_df['Date'].max()}")
    print(f"  Missing values: {test_df.isnull().sum().sum()}")
    
    return train_df, test_df


def create_features(df, is_train=True):
    """
    Create baseline time-series features.
    
    Features:
    - Temporal: day_of_week, month, year, is_weekend
    - Lag: lag_1, lag_7
    - Rolling: rolling_mean_7
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with 'Date' and 'Volume' columns
    is_train : bool
        If True, creates features for training; if False, for inference
        
    Returns:
    --------
    pd.DataFrame
        Dataframe with engineered features
    """
    df = df.copy()
    
    # Temporal features
    df['day_of_week'] = df['Date'].dt.dayofweek  # 0=Monday, 6=Sunday
    df['month'] = df['Date'].dt.month
    df['year'] = df['Date'].dt.year
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    
    # Lag features (previous values)
    df['lag_1'] = df['Volume'].shift(1)
    df['lag_7'] = df['Volume'].shift(7)
    
    # Rolling mean (7-day window)
    df['rolling_mean_7'] = df['Volume'].shift(1).rolling(window=7, min_periods=1).mean()
    
    return df


def split_train_validation(df, validation_days=90):
    """
    Split data into training and validation sets chronologically.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe with features
    validation_days : int
        Number of days to use for validation
        
    Returns:
    --------
    train_data : pd.DataFrame
        Training subset
    val_data : pd.DataFrame
        Validation subset
    """
    # Sort by date (ensure chronological order)
    df = df.sort_values('Date').reset_index(drop=True)
    
    # Split chronologically
    split_idx = len(df) - validation_days
    
    train_data = df.iloc[:split_idx].copy()
    val_data = df.iloc[split_idx:].copy()
    
    print(f"\nTraining set: {len(train_data)} samples")
    print(f"  Date range: {train_data['Date'].min()} to {train_data['Date'].max()}")
    
    print(f"\nValidation set: {len(val_data)} samples")
    print(f"  Date range: {val_data['Date'].min()} to {val_data['Date'].max()}")
    
    return train_data, val_data


def train_model(train_data, feature_cols, target_col='Volume'):
    """
    Train Random Forest Regressor.
    
    Parameters:
    -----------
    train_data : pd.DataFrame
        Training dataframe
    feature_cols : list
        List of feature column names
    target_col : str
        Target variable name
        
    Returns:
    --------
    model : RandomForestRegressor
        Trained model
    """
    X_train = train_data[feature_cols]
    y_train = train_data[target_col]
    
    # Initialize Random Forest with reasonable defaults
    model = RandomForestRegressor(
        n_estimators=100,          # Number of trees
        max_depth=15,              # Maximum depth of trees
        min_samples_split=5,       # Minimum samples to split a node
        min_samples_leaf=2,        # Minimum samples in leaf node
        random_state=RANDOM_STATE, # For reproducibility
        n_jobs=-1                  # Use all CPU cores
    )
    
    # Train the model
    print("\nTraining Random Forest model...")
    model.fit(X_train, y_train)
    print("✓ Model training complete!")
    
    return model


def evaluate_model(model, data, feature_cols, target_col='Volume', dataset_name='Validation'):
    """
    Evaluate model performance.
    
    Parameters:
    -----------
    model : RandomForestRegressor
        Trained model
    data : pd.DataFrame
        Data to evaluate on
    feature_cols : list
        Feature column names
    target_col : str
        Target variable name
    dataset_name : str
        Name of dataset (for printing)
        
    Returns:
    --------
    predictions : np.array
        Model predictions
    metrics : dict
        Dictionary of evaluation metrics
    """
    X = data[feature_cols]
    y_true = data[target_col]
    
    # Generate predictions
    y_pred = model.predict(X)
    
    # Calculate metrics
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    # Store metrics
    metrics = {
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape
    }
    
    # Print results
    print(f"\n{'='*60}")
    print(f"{dataset_name} Set Performance")
    print(f"{'='*60}")
    print(f"MAE:  {mae:,.2f}")
    print(f"RMSE: {rmse:,.2f}")
    print(f"MAPE: {mape:.2f}%")
    print(f"{'='*60}")
    
    return y_pred, metrics


def forecast_test_data(model, train_df, test_df, feature_cols):
    """
    Generate forecasts for test data.
    
    Parameters:
    -----------
    model : RandomForestRegressor
        Trained model
    train_df : pd.DataFrame
        Training data (for lag features)
    test_df : pd.DataFrame
        Test data to forecast
    feature_cols : list
        Feature column names
        
    Returns:
    --------
    test_predictions : pd.DataFrame
        Dataframe with dates and predictions
    """
    print("\n" + "="*60)
    print("STEP 5: Generating Test Forecasts")
    print("="*60)
    
    # Combine train and test for feature creation
    combined_df = pd.concat([train_df[['Date', 'Volume']], test_df[['Date', 'Volume']]], 
                            ignore_index=True)
    combined_df = combined_df.sort_values('Date').reset_index(drop=True)
    
    # Create features for combined data
    combined_features = create_features(combined_df, is_train=False)
    
    # Extract test portion
    test_start_idx = len(train_df)
    test_features = combined_features.iloc[test_start_idx:].copy()
    
    # Handle any remaining NaN values by forward filling
    test_features[feature_cols] = test_features[feature_cols].ffill()
    
    # Generate predictions
    X_test = test_features[feature_cols]
    predictions = model.predict(X_test)
    
    # Create results dataframe
    test_predictions = pd.DataFrame({
        'Date': test_features['Date'],
        'Predicted_Volume': predictions
    })
    
    print(f"\n✓ Generated {len(test_predictions)} forecasts")
    print(f"  Date range: {test_predictions['Date'].min()} to {test_predictions['Date'].max()}")
    print(f"\nPrediction Statistics:")
    print(f"  Mean:   {predictions.mean():,.0f}")
    print(f"  Median: {np.median(predictions):,.0f}")
    print(f"  Min:    {predictions.min():,.0f}")
    print(f"  Max:    {predictions.max():,.0f}")
    
    return test_predictions


def plot_results(train_data, val_data, train_pred, val_pred, model, feature_cols):
    """
    Create all visualization plots.
    
    Parameters:
    -----------
    train_data : pd.DataFrame
        Training data
    val_data : pd.DataFrame
        Validation data
    train_pred : np.array
        Training predictions
    val_pred : np.array
        Validation predictions
    model : RandomForestRegressor
        Trained model
    feature_cols : list
        Feature column names
    """
    print("\n" + "="*60)
    print("STEP 6: Creating Visualizations")
    print("="*60)
    
    # Plot 1: Train vs Validation
    fig, ax = plt.subplots(figsize=(15, 6))
    ax.plot(train_data['Date'], train_data['Volume'], 
            label='Train Actual', color='blue', alpha=0.6)
    ax.plot(train_data['Date'], train_pred, 
            label='Train Predicted', color='cyan', alpha=0.4, linestyle='--')
    ax.plot(val_data['Date'], val_data['Volume'], 
            label='Validation Actual', color='green', alpha=0.8, linewidth=2)
    ax.plot(val_data['Date'], val_pred, 
            label='Validation Predicted', color='red', alpha=0.6, linestyle='--', linewidth=2)
    ax.axvline(val_data['Date'].iloc[0], color='black', linestyle=':', alpha=0.5, label='Train/Val Split')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Passenger Volume', fontsize=12)
    ax.set_title('Train vs Validation: Actual vs Predicted', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('/mnt/user-data/outputs/plot_train_validation.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: plot_train_validation.png")
    
    # Plot 2: Validation Actual vs Predicted (Zoomed)
    fig, ax = plt.subplots(figsize=(15, 6))
    ax.plot(val_data['Date'], val_data['Volume'], label='Actual', color='blue', linewidth=2, alpha=0.7)
    ax.plot(val_data['Date'], val_pred, label='Predicted', color='red', linewidth=2, alpha=0.7, linestyle='--')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Passenger Volume', fontsize=12)
    ax.set_title('Validation Set: Actual vs Predicted', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('/mnt/user-data/outputs/plot_validation_predictions.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: plot_validation_predictions.png")
    
    # Plot 3: Residuals
    residuals = val_data['Volume'].values - val_pred
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    axes[0].scatter(val_data['Date'], residuals, alpha=0.5, color='purple')
    axes[0].axhline(y=0, color='red', linestyle='--', linewidth=2)
    axes[0].set_xlabel('Date', fontsize=12)
    axes[0].set_ylabel('Residuals', fontsize=12)
    axes[0].set_title('Residuals Over Time', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=45)
    
    axes[1].hist(residuals, bins=30, color='purple', alpha=0.7, edgecolor='black')
    axes[1].axvline(x=0, color='red', linestyle='--', linewidth=2)
    axes[1].set_xlabel('Residual Value', fontsize=12)
    axes[1].set_ylabel('Frequency', fontsize=12)
    axes[1].set_title('Residuals Distribution', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/mnt/user-data/outputs/plot_residuals.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: plot_residuals.png")
    
    # Plot 4: Feature Importance
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(range(len(importances)), importances[indices], color='steelblue', alpha=0.8)
    ax.set_xticks(range(len(importances)))
    ax.set_xticklabels([feature_cols[i] for i in indices], rotation=45, ha='right')
    ax.set_xlabel('Features', fontsize=12)
    ax.set_ylabel('Importance', fontsize=12)
    ax.set_title('Feature Importance (Random Forest)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig('/mnt/user-data/outputs/plot_feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: plot_feature_importance.png")


def main():
    """
    Main execution function.
    """
    print("\n" + "="*60)
    print("TSA PASSENGER VOLUME FORECASTING")
    print("Baseline Random Forest Model")
    print("="*60)
    
    # Step 1: Load data
    train_df, test_df = load_data(TRAIN_PATH, TEST_PATH)
    
    # Step 2: Create features
    print("\n" + "="*60)
    print("STEP 2: Feature Engineering")
    print("="*60)
    train_df_features = create_features(train_df, is_train=True)
    print("\n✓ Created features:")
    print("  - Temporal: day_of_week, month, year, is_weekend")
    print("  - Lag: lag_1, lag_7")
    print("  - Rolling: rolling_mean_7")
    
    # Step 3: Split and prepare data
    print("\n" + "="*60)
    print("STEP 3: Train-Validation Split")
    print("="*60)
    train_data, val_data = split_train_validation(train_df_features, VALIDATION_DAYS)
    
    # Define feature columns
    feature_cols = ['day_of_week', 'month', 'year', 'is_weekend', 
                    'lag_1', 'lag_7', 'rolling_mean_7']
    
    # Remove NaN from lag features
    train_data = train_data.dropna(subset=feature_cols).reset_index(drop=True)
    val_data = val_data.dropna(subset=feature_cols).reset_index(drop=True)
    print(f"\n✓ After removing NaN:")
    print(f"  Train samples: {len(train_data)}")
    print(f"  Val samples: {len(val_data)}")
    
    # Step 4: Train model
    print("\n" + "="*60)
    print("STEP 4: Model Training & Evaluation")
    print("="*60)
    model = train_model(train_data, feature_cols)
    
    # Evaluate
    train_pred, train_metrics = evaluate_model(
        model, train_data, feature_cols, dataset_name='Training'
    )
    val_pred, val_metrics = evaluate_model(
        model, val_data, feature_cols, dataset_name='Validation'
    )
    
    # Baseline comparison
    print("\n" + "="*60)
    print("Baseline Comparison (Validation Set)")
    print("="*60)
    naive_mae = mean_absolute_error(val_data['Volume'], val_data['lag_1'])
    seasonal_mae = mean_absolute_error(val_data['Volume'], val_data['lag_7'])
    print(f"Naive (lag-1) MAE:     {naive_mae:,.2f}")
    print(f"Seasonal (lag-7) MAE:  {seasonal_mae:,.2f}")
    print(f"Random Forest MAE:     {val_metrics['MAE']:,.2f}")
    print(f"\n✓ Improvement vs Naive: {(1 - val_metrics['MAE']/naive_mae)*100:.1f}%")
    print(f"✓ Improvement vs Seasonal: {(1 - val_metrics['MAE']/seasonal_mae)*100:.1f}%")
    
    # Step 5: Generate test forecasts
    test_forecasts = forecast_test_data(model, train_df, test_df, feature_cols)
    
    # Save predictions
    test_forecasts.to_csv(OUTPUT_PATH, index=False)
    print(f"\n✓ Predictions saved to: {OUTPUT_PATH}")
    
    # Step 6: Create visualizations
    plot_results(train_data, val_data, train_pred, val_pred, model, feature_cols)
    
    # Final summary
    print("\n" + "="*60)
    print("EXECUTION COMPLETE!")
    print("="*60)
    print("\nGenerated files:")
    print(f"  1. {OUTPUT_PATH}")
    print(f"  2. plot_train_validation.png")
    print(f"  3. plot_validation_predictions.png")
    print(f"  4. plot_residuals.png")
    print(f"  5. plot_feature_importance.png")
    print("\n" + "="*60 + "\n")


if __name__ == "__main__":
    main()
