"""
Bioreactor Soft Sensor Demo
---------------------------

This module simulates bioreactor process data and trains a small neural network
soft sensor to estimate glucose concentration from routinely available online
signals (Time, DO, CO2, Base addition rate).

It can be run as a script (default behavior) or imported as a module. When
imported, no training code executes automatically thanks to the main guard.

Usage (CLI):
  uv run python app.py --epochs 150 --n-samples 500 --no-show --out bioreactor_soft_sensor.png

Outputs:
  - A performance summary printed to the console
  - A figure saved to disk visualizing training, parity, residuals, etc.
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
# Heavy ML deps (torch) are imported lazily inside functions so that
# "python app.py -h" works even if PyTorch isn't installed yet.

# Set random seeds for reproducibility (torch is optional)
np.random.seed(42)
try:
    import torch  # type: ignore
    torch.manual_seed(42)
except Exception:
    pass


def generate_bioreactor_data(n_samples: int = 500) -> pd.DataFrame:
    """Generate synthetic bioreactor data.

    Simulates a fed-batch CHO culture with simple but realistic dynamics and
    returns a DataFrame with columns:
      - time_h, DO_%, CO2_%, base_rate_mL_h, glucose_g_L

    Parameters
    ----------
    n_samples: int
        Number of time points to simulate.

    Returns
    -------
    pd.DataFrame
        Simulated dataset with sensor readings and glucose.
    """
    time = np.linspace(0, 240, n_samples)  # 10 days in hours

    # Growth phases: lag -> exponential -> stationary -> decline
    cell_density = 0.3e6 * (1 + 9 * (1 / (1 + np.exp(-0.05 * (time - 48)))))
    cell_density = cell_density * (1 - 0.3 * (1 / (1 + np.exp(-0.08 * (time - 180)))))

    # Glucose: starts high, consumed over time, with feeding spikes
    glucose = 4.0 - 3.5 * (time / 240) ** 0.8
    feed_times = [72, 120, 168]  # Feed additions
    for ft in feed_times:
        glucose += 1.5 * np.exp(-((time - ft) ** 2) / 200)
    glucose = np.clip(glucose, 0.5, 6.0)

    # DO: inversely related to metabolic activity
    do = 40 + 15 * np.sin(0.02 * time) - 20 * (cell_density / cell_density.max())
    do += np.random.normal(0, 1.5, n_samples)  # Sensor noise

    # CO2: correlates with glucose consumption
    co2 = 3.0 + 2.5 * (1 - glucose / 6.0) + 0.5 * np.sin(0.03 * time)
    co2 += np.random.normal(0, 0.3, n_samples)

    # Base addition: pH control responds to lactate (proportional to glucose consumption)
    base_rate = 2.0 + 8.0 * (1 - glucose / 6.0) ** 2
    base_rate += np.random.normal(0, 0.5, n_samples)

    # Add some measurement noise to glucose
    glucose_measured = glucose + np.random.normal(0, 0.15, n_samples)

    df = pd.DataFrame({
        'time_h': time,
        'DO_%': np.clip(do, 20, 60),
        'CO2_%': np.clip(co2, 1, 6),
        'base_rate_mL_h': np.clip(base_rate, 0, 15),
        'glucose_g_L': np.clip(glucose_measured, 0.3, 6.5)
    })

    return df


def _prepare_dataloaders(
    df: pd.DataFrame,
    batch_size: int = 32,
):
    """Prepare scalers and PyTorch DataLoaders from the dataframe."""
    # Local imports to avoid hard dependency at import time
    import torch
    from torch.utils.data import TensorDataset, DataLoader
    # Prepare features and target
    X = df[['time_h', 'DO_%', 'CO2_%', 'base_rate_mL_h']].values
    y = df['glucose_g_L'].values.reshape(-1, 1)

    # Split data: 70% train, 15% validation, 15% test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.176, random_state=42
    )

    print(f"\nTraining samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples: {len(X_test)}")

    # Scale features
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_train_scaled = scaler_X.fit_transform(X_train)
    X_val_scaled = scaler_X.transform(X_val)
    X_test_scaled = scaler_X.transform(X_test)

    y_train_scaled = scaler_y.fit_transform(y_train)
    y_val_scaled = scaler_y.transform(y_val)
    y_test_scaled = scaler_y.transform(y_test)

    # Convert to PyTorch tensors
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train_scaled),
        torch.FloatTensor(y_train_scaled)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val_scaled),
        torch.FloatTensor(y_val_scaled)
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return (
        (X_train, X_val, X_test,
         y_train, y_val, y_test,
         X_train_scaled, X_val_scaled, X_test_scaled,
         scaler_X, scaler_y,
         train_loader, val_loader)
    )


def _train_model(train_loader, val_loader, n_epochs: int = 150):
    """Train the neural network and return the model and loss histories."""
    import torch
    import torch.nn as nn

    # Define neural network locally to avoid torch at module import time
    class GlucoseSoftSensor(nn.Module):
        def __init__(self, input_dim=4):
            super(GlucoseSoftSensor, self).__init__()
            self.network = nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(32, 16),
                nn.ReLU(),
                nn.Linear(16, 1)
            )

        def forward(self, x):
            return self.network(x)

    model = GlucoseSoftSensor(input_dim=4)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    train_losses = []
    val_losses = []

    print("\nTraining neural network soft sensor...")
    for epoch in range(n_epochs):
        # Training
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                y_pred = model(X_batch)
                loss = criterion(y_pred, y_batch)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        if (epoch + 1) % 30 == 0:
            print(
                f"Epoch [{epoch + 1}/{n_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
            )

    print("\nTraining complete!")
    return model, train_losses, val_losses

def _visualize_and_report(
    model,
    scaler_X,
    scaler_y,
    X_train,
    X_val,
    X_test,
    y_train,
    y_val,
    y_test,
    X_test_scaled,
    train_losses,
    val_losses,
    out_path: str = 'bioreactor_soft_sensor.png',
    show: bool = True,
):
    """Create visualizations, print metrics, and optionally save/show the figure."""
    import torch
    # Test set predictions
    model.eval()
    with torch.no_grad():
        y_test_pred_scaled = model(torch.FloatTensor(X_test_scaled)).numpy()

    # Inverse transform predictions
    y_test_pred = scaler_y.inverse_transform(y_test_pred_scaled)

    # Calculate metrics
    mse = mean_squared_error(y_test, y_test_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_test_pred)

    print(f"\nTest Set Performance:")
    print(f"RMSE: {rmse:.3f} g/L")
    print(f"R²: {r2:.4f}")

    # Create comprehensive visualizations
    fig = plt.figure(figsize=(16, 10))

    # 1. Training history
    ax1 = plt.subplot(2, 3, 1)
    ax1.plot(train_losses, label='Training Loss', linewidth=2, alpha=0.8)
    ax1.plot(val_losses, label='Validation Loss', linewidth=2, alpha=0.8)
    ax1.set_xlabel('Epoch', fontsize=11, fontweight='bold')
    ax1.set_ylabel('MSE Loss', fontsize=11, fontweight='bold')
    ax1.set_title('A. Model Training Progress', fontsize=12, fontweight='bold', pad=10)
    ax1.legend(frameon=True, shadow=True)
    ax1.grid(True, alpha=0.3)

    # 2. Parity plot
    ax2 = plt.subplot(2, 3, 2)
    ax2.scatter(y_test, y_test_pred, alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
    min_val = min(y_test.min(), y_test_pred.min())
    max_val = max(y_test.max(), y_test_pred.max())
    ax2.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Ideal')
    ax2.set_xlabel('Measured Glucose (g/L)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Predicted Glucose (g/L)', fontsize=11, fontweight='bold')
    ax2.set_title(f'B. Parity Plot (R² = {r2:.4f})', fontsize=12, fontweight='bold', pad=10)
    ax2.legend(frameon=True, shadow=True)
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal', adjustable='box')

    # 3. Residuals
    ax3 = plt.subplot(2, 3, 3)
    residuals = y_test - y_test_pred
    ax3.scatter(y_test_pred, residuals, alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
    ax3.axhline(y=0, color='r', linestyle='--', linewidth=2)
    ax3.set_xlabel('Predicted Glucose (g/L)', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Residuals (g/L)', fontsize=11, fontweight='bold')
    ax3.set_title(f'C. Residual Analysis (RMSE = {rmse:.3f} g/L)', fontsize=12, fontweight='bold', pad=10)
    ax3.grid(True, alpha=0.3)

    # 4. Time series reconstruction (test set)
    test_indices = np.argsort(X_test[:, 0])  # Sort by time
    ax4 = plt.subplot(2, 3, 4)
    ax4.plot(
        X_test[test_indices, 0], y_test[test_indices], 'o-',
        label='Measured (offline)', linewidth=2, markersize=6, alpha=0.7
    )
    ax4.plot(
        X_test[test_indices, 0], y_test_pred[test_indices], 's-',
        label='Soft Sensor (real-time)', linewidth=2, markersize=5, alpha=0.7
    )
    ax4.set_xlabel('Time (hours)', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Glucose (g/L)', fontsize=11, fontweight='bold')
    ax4.set_title('D. Real-Time Prediction on Test Batch', fontsize=12, fontweight='bold', pad=10)
    ax4.legend(frameon=True, shadow=True)
    ax4.grid(True, alpha=0.3)

    # 5. Feature importance (simple sensitivity analysis)
    ax5 = plt.subplot(2, 3, 5)
    feature_names = ['Time', 'DO', 'CO₂', 'Base Rate']
    # Vary each feature ±10% around mean and measure output change
    baseline = scaler_X.transform(X_test.mean(axis=0).reshape(1, -1))
    sensitivities = []
    for i in range(4):
        perturbed_up = baseline.copy()
        perturbed_up[0, i] *= 1.1
        perturbed_down = baseline.copy()
        perturbed_down[0, i] *= 0.9

        with torch.no_grad():
            pred_up = scaler_y.inverse_transform(
                model(torch.FloatTensor(perturbed_up)).numpy()
            )[0, 0]
            pred_down = scaler_y.inverse_transform(
                model(torch.FloatTensor(perturbed_down)).numpy()
            )[0, 0]

        sensitivity = abs(pred_up - pred_down)
        sensitivities.append(sensitivity)

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    ax5.barh(feature_names, sensitivities, color=colors, edgecolor='black', linewidth=1.5)
    ax5.set_xlabel('Sensitivity (Δ Glucose, g/L)', fontsize=11, fontweight='bold')
    ax5.set_title('E. Feature Sensitivity Analysis', fontsize=12, fontweight='bold', pad=10)
    ax5.grid(True, alpha=0.3, axis='x')

    # 6. Error distribution
    ax6 = plt.subplot(2, 3, 6)
    ax6.hist(residuals, bins=20, edgecolor='black', alpha=0.7, color='steelblue')
    ax6.axvline(x=0, color='r', linestyle='--', linewidth=2)
    ax6.set_xlabel('Prediction Error (g/L)', fontsize=11, fontweight='bold')
    ax6.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax6.set_title('F. Error Distribution', fontsize=12, fontweight='bold', pad=10)
    ax6.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        print(f"\nFigure saved as '{out_path}'")
    if show:
        plt.show()
    else:
        plt.close(fig)

    # Print summary table
    print("\n" + "=" * 60)
    print("SOFT SENSOR PERFORMANCE SUMMARY")
    print("=" * 60)
    print(f"Model Architecture: 4 → 64 → 32 → 16 → 1")
    print(f"Training Samples: {len(X_train)}")
    print(f"Test RMSE: {rmse:.3f} g/L")
    print(f"Test R²: {r2:.4f}")
    print(f"Typical measurement uncertainty: ±0.15 g/L")
    print(f"Model prediction uncertainty: ±{rmse:.3f} g/L")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Train a soft sensor on synthetic bioreactor data.")
    parser.add_argument("--n-samples", type=int, default=500, help="Number of samples to simulate (default: 500)")
    parser.add_argument("--epochs", type=int, default=150, help="Training epochs (default: 150)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size (default: 32)")
    parser.add_argument("--out", type=str, default="bioreactor_soft_sensor.png", help="Output figure path")
    parser.add_argument("--no-show", action="store_true", help="Do not display the plot window")
    args = parser.parse_args()

    # Generate data
    df = generate_bioreactor_data(n_samples=args.n_samples)
    print("Dataset shape:", df.shape)
    print("\nFirst few rows:")
    print(df.head())
    print("\nBasic statistics:")
    print(df.describe())

    (
        X_train, X_val, X_test,
        y_train, y_val, y_test,
        X_train_scaled, X_val_scaled, X_test_scaled,
        scaler_X, scaler_y,
        train_loader, val_loader
    ) = _prepare_dataloaders(df, batch_size=args.batch_size)

    model, train_losses, val_losses = _train_model(train_loader, val_loader, n_epochs=args.epochs)

    _visualize_and_report(
        model,
        scaler_X,
        scaler_y,
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        X_test_scaled,
        train_losses,
        val_losses,
        out_path=args.out,
        show=not args.no_show,
    )


if __name__ == "__main__":
    main()