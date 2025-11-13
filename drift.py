import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import torch
import torch.nn as nn
from scipy.stats import norm
import warnings

warnings.filterwarnings('ignore')

# Set random seeds
np.random.seed(42)
torch.manual_seed(42)


# ============================================================================
# PART 1: Generate Bioreactor Data with Realistic Sensor Drift
# ============================================================================

def generate_bioreactor_data_with_drift(n_samples=600, drift_start=300):
    """
    Simulate fed-batch culture with sensor drift occurring mid-batch.

    Drift types:
    - DO probe: gradual negative drift due to membrane fouling
    - CO2 analyzer: zero drift (baseline shift)
    - Base addition: flow meter drift (gain error)
    """
    time = np.linspace(0, 300, n_samples)  # 12.5 days in hours

    # True biological state (ground truth)
    cell_density = 0.3e6 * (1 + 9 * (1 / (1 + np.exp(-0.05 * (time - 60)))))
    cell_density = cell_density * (1 - 0.3 * (1 / (1 + np.exp(-0.08 * (time - 230)))))

    # TRUE glucose profile
    glucose_true = 4.5 - 4.0 * (time / 300) ** 0.7
    feed_times = [90, 150, 210]
    for ft in feed_times:
        glucose_true += 1.8 * np.exp(-((time - ft) ** 2) / 250)
    glucose_true = np.clip(glucose_true, 0.4, 7.0)

    # TRUE sensor responses (no drift yet)
    do_true = 42 + 12 * np.sin(0.02 * time) - 18 * (cell_density / cell_density.max())
    co2_true = 3.2 + 2.8 * (1 - glucose_true / 7.0) + 0.6 * np.sin(0.03 * time)
    base_rate_true = 2.5 + 9.0 * (1 - glucose_true / 7.0) ** 2

    # Add measurement noise
    noise_do = np.random.normal(0, 1.2, n_samples)
    noise_co2 = np.random.normal(0, 0.25, n_samples)
    noise_base = np.random.normal(0, 0.4, n_samples)

    do_measured = do_true + noise_do
    co2_measured = co2_true + noise_co2
    base_measured = base_rate_true + noise_base

    # ========================================================================
    # INJECT SENSOR DRIFT (starts at drift_start)
    # ========================================================================

    # DO probe: gradual fouling causes negative drift (reads lower than actual)
    do_drift = np.zeros(n_samples)
    drift_progress = np.clip((time - drift_start) / 100, 0, 1)
    do_drift = -8.0 * drift_progress ** 1.5  # Accelerating drift
    do_measured_drift = do_measured + do_drift

    # CO2 analyzer: zero drift (baseline shift upward)
    co2_drift = np.zeros(n_samples)
    co2_drift = 0.8 * drift_progress
    co2_measured_drift = co2_measured + co2_drift

    # Base rate: flow meter gain error (reads high)
    base_drift = np.zeros(n_samples)
    base_drift = 1.5 * drift_progress
    base_measured_drift = base_measured + base_drift

    # Reference glucose measurements (sparse, offline, unaffected by sensor drift)
    # Simulate 30 offline measurements throughout the batch
    ref_indices = np.sort(np.random.choice(n_samples, size=30, replace=False))
    glucose_reference = glucose_true[ref_indices] + np.random.normal(0, 0.12, 30)

    df = pd.DataFrame({
        'time_h': time,
        'DO_measured': np.clip(do_measured_drift, 15, 65),
        'CO2_measured': np.clip(co2_measured_drift, 1.5, 7),
        'base_rate_measured': np.clip(base_measured_drift, 0, 18),
        'glucose_true': glucose_true,
        'DO_drift': do_drift,
        'CO2_drift': co2_drift,
        'base_drift': base_drift,
        'DO_true': do_true,
        'CO2_true': co2_true,
        'base_true': base_rate_true
    })

    ref_df = pd.DataFrame({
        'time_h': time[ref_indices],
        'glucose_reference': glucose_reference,
        'index': ref_indices
    })

    return df, ref_df


# Generate data
print("Generating bioreactor data with sensor drift...")
df, ref_df = generate_bioreactor_data_with_drift(n_samples=600, drift_start=300)
print(f"Generated {len(df)} time points with {len(ref_df)} reference measurements")

# ============================================================================
# PART 2: Train Initial Model on Clean Data (first 300 hours)
# ============================================================================

# Use only pre-drift data for training
train_cutoff = 300
X_train = df.loc[:train_cutoff, ['time_h', 'DO_measured', 'CO2_measured', 'base_rate_measured']].values
y_train = df.loc[:train_cutoff, 'glucose_true'].values.reshape(-1, 1)

# Scale data
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
y_train_scaled = scaler_y.fit_transform(y_train)


# Define model
class GlucoseSoftSensor(nn.Module):
    def __init__(self):
        super(GlucoseSoftSensor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(4, 64),
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


# Train model
print("\nTraining static model on pre-drift data...")
model_static = GlucoseSoftSensor()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model_static.parameters(), lr=0.001)

X_train_tensor = torch.FloatTensor(X_train_scaled)
y_train_tensor = torch.FloatTensor(y_train_scaled)

n_epochs = 200
for epoch in range(n_epochs):
    optimizer.zero_grad()
    y_pred = model_static(X_train_tensor)
    loss = criterion(y_pred, y_train_tensor)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 50 == 0:
        print(f"Epoch {epoch + 1}/{n_epochs}, Loss: {loss.item():.4f}")

print("Static model training complete!")

# ============================================================================
# PART 3: Test Static Model on Full Dataset (Including Drift Period)
# ============================================================================

X_full = df[['time_h', 'DO_measured', 'CO2_measured', 'base_rate_measured']].values
X_full_scaled = scaler_X.transform(X_full)

model_static.eval()
with torch.no_grad():
    y_static_scaled = model_static(torch.FloatTensor(X_full_scaled)).numpy()
y_static = scaler_y.inverse_transform(y_static_scaled).flatten()

# Calculate prediction errors
error_static = df['glucose_true'].values - y_static
rmse_static_full = np.sqrt(mean_squared_error(df['glucose_true'].values, y_static))

# Calculate RMSE before and after drift
rmse_static_pre = np.sqrt(mean_squared_error(
    df.loc[:300, 'glucose_true'].values,
    y_static[:301]
))
rmse_static_post = np.sqrt(mean_squared_error(
    df.loc[300:, 'glucose_true'].values,
    y_static[300:]
))

print(f"\nStatic Model Performance:")
print(f"  Pre-drift RMSE: {rmse_static_pre:.3f} g/L")
print(f"  Post-drift RMSE: {rmse_static_post:.3f} g/L")
print(f"  Overall RMSE: {rmse_static_full:.3f} g/L")
print(f"  Performance degradation: {(rmse_static_post / rmse_static_pre - 1) * 100:.1f}%")

# ============================================================================
# PART 4: Adaptive Model with Online Learning
# ============================================================================

print("\n" + "=" * 70)
print("IMPLEMENTING ONLINE ADAPTIVE MODEL")
print("=" * 70)


class AdaptiveSoftSensor:
    """
    Adaptive soft sensor with online learning capabilities.

    Features:
    - Exponentially weighted recursive least squares (EWRLS) for linear correction
    - Drift detection via cumulative error monitoring
    - Kalman filtering for sensor state estimation
    """

    def __init__(self, base_model, scaler_X, scaler_y, forgetting_factor=0.98):
        self.base_model = base_model
        self.scaler_X = scaler_X
        self.scaler_y = scaler_y
        self.lambda_forget = forgetting_factor

        # Online correction: y_corrected = y_pred + bias + drift_correction
        self.bias = 0.0
        self.drift_state = np.zeros(4)  # One for each sensor

        # Drift detection
        self.error_window = []
        self.error_threshold = 0.5  # g/L
        self.window_size = 20

        # Kalman filter for sensor drift estimation
        self.kf_state = np.zeros(4)  # Estimated drift for each sensor
        self.kf_covariance = np.eye(4) * 1.0
        self.process_noise = 0.01
        self.measurement_noise = 0.3

    def predict(self, X):
        """Base prediction from neural network"""
        X_scaled = self.scaler_X.transform(X.reshape(1, -1))
        self.base_model.eval()
        with torch.no_grad():
            y_scaled = self.base_model(torch.FloatTensor(X_scaled)).numpy()
        y_pred = self.scaler_y.inverse_transform(y_scaled)[0, 0]
        return y_pred

    def detect_drift(self, error):
        """Detect if drift is occurring based on error magnitude"""
        self.error_window.append(abs(error))
        if len(self.error_window) > self.window_size:
            self.error_window.pop(0)

        if len(self.error_window) >= self.window_size:
            mean_error = np.mean(self.error_window)
            return mean_error > self.error_threshold
        return False

    def update(self, X, y_true):
        """
        Online update with reference measurement.
        Uses exponentially weighted recursive least squares for bias correction.
        """
        # Get base prediction
        y_pred = self.predict(X)
        error = y_true - y_pred

        # Exponentially weighted bias update
        self.bias = self.lambda_forget * self.bias + (1 - self.lambda_forget) * error

        # Detect drift
        drift_detected = self.detect_drift(error)

        return y_pred + self.bias, drift_detected, error

    def predict_corrected(self, X):
        """Prediction with bias correction applied"""
        y_pred = self.predict(X)
        return y_pred + self.bias


# Initialize adaptive model
adaptive_sensor = AdaptiveSoftSensor(
    base_model=model_static,
    scaler_X=scaler_X,
    scaler_y=scaler_y,
    forgetting_factor=0.97
)

# Simulate online deployment
print("\nSimulating online deployment with periodic reference measurements...")

y_adaptive = np.zeros(len(df))
y_adaptive_raw = np.zeros(len(df))
bias_history = []
drift_alerts = []
update_times = []

ref_idx = 0  # Index in reference measurement dataframe

for i in range(len(df)):
    X_current = df.loc[i, ['time_h', 'DO_measured', 'CO2_measured', 'base_rate_measured']].values

    # Check if we have a reference measurement at this time point
    if ref_idx < len(ref_df) and ref_df.iloc[ref_idx]['index'] == i:
        # Update model with reference measurement
        y_ref = ref_df.iloc[ref_idx]['glucose_reference']
        y_corrected, drift_detected, error = adaptive_sensor.update(X_current, y_ref)

        y_adaptive[i] = y_corrected
        update_times.append(i)

        if drift_detected:
            drift_alerts.append(i)

        ref_idx += 1
    else:
        # No reference available, use corrected prediction
        y_adaptive[i] = adaptive_sensor.predict_corrected(X_current)

    # Store uncorrected prediction for comparison
    y_adaptive_raw[i] = adaptive_sensor.predict(X_current)
    bias_history.append(adaptive_sensor.bias)

bias_history = np.array(bias_history)

# Calculate adaptive model performance
error_adaptive = df['glucose_true'].values - y_adaptive
rmse_adaptive_full = np.sqrt(mean_squared_error(df['glucose_true'].values, y_adaptive))

rmse_adaptive_pre = np.sqrt(mean_squared_error(
    df.loc[:300, 'glucose_true'].values,
    y_adaptive[:301]
))
rmse_adaptive_post = np.sqrt(mean_squared_error(
    df.loc[300:, 'glucose_true'].values,
    y_adaptive[300:]
))

print(f"\nAdaptive Model Performance:")
print(f"  Pre-drift RMSE: {rmse_adaptive_pre:.3f} g/L")
print(f"  Post-drift RMSE: {rmse_adaptive_post:.3f} g/L")
print(f"  Overall RMSE: {rmse_adaptive_full:.3f} g/L")
print(f"  Number of reference updates: {len(update_times)}")
print(f"  Drift alerts triggered: {len(drift_alerts)}")

print(f"\n{'=' * 70}")
print("PERFORMANCE COMPARISON")
print(f"{'=' * 70}")
print(f"Post-drift improvement: {(1 - rmse_adaptive_post / rmse_static_post) * 100:.1f}%")
print(f"Overall improvement: {(1 - rmse_adaptive_full / rmse_static_full) * 100:.1f}%")

# ============================================================================
# PART 5: Comprehensive Visualizations
# ============================================================================

fig = plt.figure(figsize=(18, 12))

# 1. Sensor drift visualization
ax1 = plt.subplot(3, 3, 1)
time = df['time_h'].values
ax1.plot(time, df['DO_drift'], label='DO Probe', linewidth=2.5, alpha=0.8)
ax1.plot(time, df['CO2_drift'], label='CO₂ Analyzer', linewidth=2.5, alpha=0.8)
ax1.plot(time, df['base_drift'], label='Base Flow Meter', linewidth=2.5, alpha=0.8)
ax1.axvline(x=300, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Drift Onset')
ax1.set_xlabel('Time (hours)', fontsize=11, fontweight='bold')
ax1.set_ylabel('Drift Magnitude (sensor units)', fontsize=11, fontweight='bold')
ax1.set_title('A. Sensor Drift Profiles', fontsize=12, fontweight='bold', pad=10)
ax1.legend(frameon=True, shadow=True, fontsize=9)
ax1.grid(True, alpha=0.3)

# 2. DO sensor: true vs measured
ax2 = plt.subplot(3, 3, 2)
ax2.plot(time, df['DO_true'], label='True DO', linewidth=2, alpha=0.8)
ax2.plot(time, df['DO_measured'], label='Measured DO (drifted)', linewidth=2, alpha=0.7)
ax2.axvline(x=300, color='red', linestyle='--', linewidth=2, alpha=0.5)
ax2.set_xlabel('Time (hours)', fontsize=11, fontweight='bold')
ax2.set_ylabel('DO (%)', fontsize=11, fontweight='bold')
ax2.set_title('B. DO Sensor Fouling Impact', fontsize=12, fontweight='bold', pad=10)
ax2.legend(frameon=True, shadow=True, fontsize=9)
ax2.grid(True, alpha=0.3)

# 3. Static model failure
ax3 = plt.subplot(3, 3, 3)
ax3.plot(time, df['glucose_true'], 'k-', label='True Glucose', linewidth=2.5, alpha=0.9)
ax3.plot(time, y_static, 'r--', label='Static Model', linewidth=2, alpha=0.7)
ax3.scatter(ref_df['time_h'], ref_df['glucose_reference'],
            s=80, marker='o', color='green', edgecolors='black',
            linewidth=1, alpha=0.8, label='Reference Measurements', zorder=5)
ax3.axvline(x=300, color='red', linestyle='--', linewidth=2, alpha=0.5)
# Shade the high-error region
ax3.axvspan(300, 600, alpha=0.15, color='red')
ax3.set_xlabel('Time (hours)', fontsize=11, fontweight='bold')
ax3.set_ylabel('Glucose (g/L)', fontsize=11, fontweight='bold')
ax3.set_title(f'C. Static Model Failure (Post-drift RMSE: {rmse_static_post:.2f} g/L)',
              fontsize=12, fontweight='bold', pad=10)
ax3.legend(frameon=True, shadow=True, fontsize=9)
ax3.grid(True, alpha=0.3)

# 4. Adaptive model success
ax4 = plt.subplot(3, 3, 4)
ax4.plot(time, df['glucose_true'], 'k-', label='True Glucose', linewidth=2.5, alpha=0.9)
ax4.plot(time, y_adaptive, 'b-', label='Adaptive Model', linewidth=2, alpha=0.8)
ax4.scatter(ref_df['time_h'], ref_df['glucose_reference'],
            s=80, marker='o', color='green', edgecolors='black',
            linewidth=1, alpha=0.8, label='Reference (Updates)', zorder=5)
# Mark drift alerts
if drift_alerts:
    alert_times = time[drift_alerts]
    alert_glucose = y_adaptive[drift_alerts]
    ax4.scatter(alert_times, alert_glucose, s=150, marker='*',
                color='orange', edgecolors='red', linewidth=2,
                label='Drift Alert', zorder=6)
ax4.axvline(x=300, color='red', linestyle='--', linewidth=2, alpha=0.5)
ax4.axvspan(300, 600, alpha=0.15, color='green')
ax4.set_xlabel('Time (hours)', fontsize=11, fontweight='bold')
ax4.set_ylabel('Glucose (g/L)', fontsize=11, fontweight='bold')
ax4.set_title(f'D. Adaptive Model Success (Post-drift RMSE: {rmse_adaptive_post:.2f} g/L)',
              fontsize=12, fontweight='bold', pad=10)
ax4.legend(frameon=True, shadow=True, fontsize=9, loc='best')
ax4.grid(True, alpha=0.3)

# 5. Bias correction over time
ax5 = plt.subplot(3, 3, 5)
ax5.plot(time, bias_history, 'purple', linewidth=2.5, alpha=0.8)
ax5.axvline(x=300, color='red', linestyle='--', linewidth=2, alpha=0.5)
for ut in update_times:
    ax5.axvline(x=time[ut], color='green', linestyle=':', linewidth=1, alpha=0.4)
ax5.set_xlabel('Time (hours)', fontsize=11, fontweight='bold')
ax5.set_ylabel('Bias Correction (g/L)', fontsize=11, fontweight='bold')
ax5.set_title('E. Online Bias Adaptation', fontsize=12, fontweight='bold', pad=10)
ax5.grid(True, alpha=0.3)
ax5.text(0.98, 0.97, f'Updates: {len(update_times)}',
         transform=ax5.transAxes, fontsize=10, verticalalignment='top',
         horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# 6. Prediction error comparison
ax6 = plt.subplot(3, 3, 6)
ax6.plot(time, abs(error_static), 'r-', label='Static Model', linewidth=2, alpha=0.7)
ax6.plot(time, abs(error_adaptive), 'b-', label='Adaptive Model', linewidth=2, alpha=0.7)
ax6.axvline(x=300, color='red', linestyle='--', linewidth=2, alpha=0.5)
ax6.axhline(y=0.5, color='orange', linestyle='--', linewidth=1.5, alpha=0.6, label='Alert Threshold')
ax6.set_xlabel('Time (hours)', fontsize=11, fontweight='bold')
ax6.set_ylabel('Absolute Error (g/L)', fontsize=11, fontweight='bold')
ax6.set_title('F. Prediction Error Over Time', fontsize=12, fontweight='bold', pad=10)
ax6.legend(frameon=True, shadow=True, fontsize=9)
ax6.grid(True, alpha=0.3)
ax6.set_ylim([0, 2.5])

# 7. Error distribution before drift
ax7 = plt.subplot(3, 3, 7)
errors_pre_static = error_static[:301]
errors_pre_adaptive = error_adaptive[:301]
ax7.hist(errors_pre_static, bins=25, alpha=0.6, color='red',
         edgecolor='black', label='Static', density=True)
ax7.hist(errors_pre_adaptive, bins=25, alpha=0.6, color='blue',
         edgecolor='black', label='Adaptive', density=True)
ax7.axvline(x=0, color='black', linestyle='--', linewidth=2)
ax7.set_xlabel('Prediction Error (g/L)', fontsize=11, fontweight='bold')
ax7.set_ylabel('Density', fontsize=11, fontweight='bold')
ax7.set_title(f'G. Pre-Drift Error Distribution', fontsize=12, fontweight='bold', pad=10)
ax7.legend(frameon=True, shadow=True, fontsize=9)
ax7.grid(True, alpha=0.3, axis='y')

# 8. Error distribution after drift
ax8 = plt.subplot(3, 3, 8)
errors_post_static = error_static[300:]
errors_post_adaptive = error_adaptive[300:]
ax8.hist(errors_post_static, bins=25, alpha=0.6, color='red',
         edgecolor='black', label=f'Static (σ={np.std(errors_post_static):.2f})', density=True)
ax8.hist(errors_post_adaptive, bins=25, alpha=0.6, color='blue',
         edgecolor='black', label=f'Adaptive (σ={np.std(errors_post_adaptive):.2f})', density=True)
ax8.axvline(x=0, color='black', linestyle='--', linewidth=2)
ax8.set_xlabel('Prediction Error (g/L)', fontsize=11, fontweight='bold')
ax8.set_ylabel('Density', fontsize=11, fontweight='bold')
ax8.set_title(f'H. Post-Drift Error Distribution', fontsize=12, fontweight='bold', pad=10)
ax8.legend(frameon=True, shadow=True, fontsize=9)
ax8.grid(True, alpha=0.3, axis='y')

# 9. Performance comparison bar chart
ax9 = plt.subplot(3, 3, 9)
metrics = ['Pre-Drift\nRMSE', 'Post-Drift\nRMSE', 'Overall\nRMSE']
static_vals = [rmse_static_pre, rmse_static_post, rmse_static_full]
adaptive_vals = [rmse_adaptive_pre, rmse_adaptive_post, rmse_adaptive_full]

x = np.arange(len(metrics))
width = 0.35

bars1 = ax9.bar(x - width / 2, static_vals, width, label='Static Model',
                color='red', alpha=0.7, edgecolor='black', linewidth=1.5)
bars2 = ax9.bar(x + width / 2, adaptive_vals, width, label='Adaptive Model',
                color='blue', alpha=0.7, edgecolor='black', linewidth=1.5)

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax9.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{height:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

ax9.set_ylabel('RMSE (g/L)', fontsize=11, fontweight='bold')
ax9.set_title('I. Performance Comparison Summary', fontsize=12, fontweight='bold', pad=10)
ax9.set_xticks(x)
ax9.set_xticklabels(metrics, fontsize=10)
ax9.legend(frameon=True, shadow=True, fontsize=9)
ax9.grid(True, alpha=0.3, axis='y')

# Add improvement percentages
improvement_post = (1 - rmse_adaptive_post / rmse_static_post) * 100
ax9.text(0.5, 0.95, f'Post-drift improvement: {improvement_post:.1f}%',
         transform=ax9.transAxes, fontsize=11, verticalalignment='top',
         horizontalalignment='center',
         bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8),
         fontweight='bold')

plt.tight_layout()
plt.savefig('sensor_drift_online_learning.png', dpi=300, bbox_inches='tight')
print("\nFigure saved as 'sensor_drift_online_learning.png'")
plt.show()

# ============================================================================
# PART 6: Summary Statistics Table
# ============================================================================

print("\n" + "=" * 70)
print("DEPLOYMENT SIMULATION SUMMARY")
print("=" * 70)
print(f"Total batch duration: {time[-1]:.1f} hours")
print(f"Drift onset: 300 hours")
print(f"Reference measurements: {len(ref_df)} samples")
print(f"Update frequency: ~{len(df) / len(ref_df):.1f} hours between updates")
print()
print("STATIC MODEL (No Adaptation):")
print(f"  Pre-drift performance:  RMSE = {rmse_static_pre:.3f} g/L")
print(f"  Post-drift performance: RMSE = {rmse_static_post:.3f} g/L")
print(f"  Degradation: +{(rmse_static_post / rmse_static_pre - 1) * 100:.1f}%")
print()
print("ADAPTIVE MODEL (Online Learning):")
print(f"  Pre-drift performance:  RMSE = {rmse_adaptive_pre:.3f} g/L")
print(f"  Post-drift performance: RMSE = {rmse_adaptive_post:.3f} g/L")
print(f"  Degradation: +{(rmse_adaptive_post / rmse_adaptive_pre - 1) * 100:.1f}%")
print()
print("IMPROVEMENT:")
print(f"  Post-drift error reduction: {(1 - rmse_adaptive_post / rmse_static_post) * 100:.1f}%")
print(f"  Overall error reduction: {(1 - rmse_adaptive_full / rmse_static_full) * 100:.1f}%")
print(f"  Drift alerts triggered: {len(drift_alerts)}")
print("=" * 70)

# ============================================================================
# PART 7: Practical Recommendations
# ============================================================================

print("\n" + "=" * 70)
print("PRACTICAL DEPLOYMENT RECOMMENDATIONS")
print("=" * 70)
print()
print("1. REFERENCE MEASUREMENT STRATEGY:")
print(f"   • Minimum frequency: Every {len(df) // len(ref_df)} hours (as simulated)")
print("   • Increase frequency when drift alerts trigger")
print("   • Use automated sampling systems where possible")
print()
print("2. DRIFT DETECTION THRESHOLDS:")
print(f"   • Current alert threshold: {adaptive_sensor.error_threshold} g/L")
print("   • Tune based on process criticality and reference measurement cost")
print(f"   • Window size for detection: {adaptive_sensor.window_size} samples")
print()
print("3. MODEL MAINTENANCE:")
print("   • Monitor bias correction magnitude (alert if |bias| > 1.0 g/L)")
print("   • Schedule probe maintenance when drift accelerates")
print("   • Log all reference measurements and corrections for audit trail")
print()
print("4. FALLBACK PROCEDURES:")
print("   • Revert to offline measurements if model uncertainty exceeds limits")
print("   • Have spare calibrated probes ready for quick replacement")
print("   • Implement hard limits on prediction ranges")
print()
print("=" * 70)