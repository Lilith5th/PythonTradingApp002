import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Sample test for log transformation
def test_log_transformation():
    # Create sample price data
    dates = pd.date_range(start='2020-01-01', periods=100)
    # Exponential growth price data (good candidate for log transform)
    prices = 100 * np.exp(0.01 * np.arange(100)) + np.random.normal(0, 2, 100)
    
    df = pd.DataFrame({
        'date': dates,
        'price': prices
    })
    
    # Plot original prices
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.plot(df['date'], df['price'])
    plt.title('Original Price')
    plt.grid(True)
    
    # Apply log transformation
    df['log_price'] = np.log(df['price'])
    
    plt.subplot(2, 2, 2)
    plt.plot(df['date'], df['log_price'])
    plt.title('Log-Transformed Price')
    plt.grid(True)
    
    # Plot returns
    df['returns'] = df['price'].pct_change()
    df['log_returns'] = df['log_price'].diff()
    
    plt.subplot(2, 2, 3)
    plt.plot(df['date'][1:], df['returns'][1:])
    plt.title('Returns')
    plt.grid(True)
    
    plt.subplot(2, 2, 4)
    plt.plot(df['date'][1:], df['log_returns'][1:])
    plt.title('Log Returns')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Test inverse transformation
    log_transformed_values = np.log(prices)
    inverse_transformed = np.exp(log_transformed_values)
    
    max_diff = np.max(np.abs(prices - inverse_transformed))
    print(f"Maximum difference after round-trip transformation: {max_diff:.8f}")
    
    # This should be very close to zero, confirming the transformation is reversible

# Run the test
if __name__ == "__main__":
    test_log_transformation()