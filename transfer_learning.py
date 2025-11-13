import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from scipy.stats import pearsonr
import warnings

warnings.filterwarnings('ignore')

# Set random seeds
np.random.seed(42)
torch.manual_seed(42)


# ============================================================================
# PART 1: Generate Multi-Scale Bioreactor Data
# ============================================================================

def generate_batch_data(n_batches, scale_name, scale_volume, base_kla, mixing_time):
    """
    Generate bioreactor batches at different scales.

    Key scale effects:
    - kLa: decreases with scale (oxygen transfer limitation)
    - Mixing time: increases with scale (creates gradients)
    - Temperature control: harder at scale (thermal lag)
    - Cell response: affected by heterogeneity at large scale
    """
    all_batches = []

    for batch_id in range(n_batches):
        n_samples = 400  # ~10 days, 36 min intervals
        time = np.linspace(0, 240, n_samples)

        # Batch-to-batch variability (media lot, inoculum, etc.)
        batch_variability = np.random.uniform(0.85, 1.15)

        # Cell growth (logistic with scale-dependent max density)
        # Larger scales: slightly lower max density due to heterogeneity
        scale_factor = 1.0 - 0.15 * (scale_volume / 10000)  # Up to 15% reduction
        max_density = 8.0e6 * scale_factor * batch_variability
        cell_density = 0.5e6 * max_density / (0.5e6 + (max_density - 0.5e6) * np.exp(-0.04 * time))

        # Death phase (more pronounced at larger scale due to shear/heterogeneity)
        death_intensity = 0.2 + 0.15 * (scale_volume / 10000)
        cell_density *= (1 - death_intensity / (1 + np.exp(-0.1 * (time - 200))))

        # Glucose consumption (faster at bench due to better mixing)
        glucose = 5.0 * batch_variability - 4.5 * (time / 240) ** 0.8

        # Feed additions (timing varies batch-to-batch)
        feed_variation = np.random.uniform(-10, 10)
        feed_times = [80 + feed_variation, 140 + feed_variation, 190 + feed_variation]
        for ft in feed_times:
            glucose += 2.0 * np.exp(-((time - ft) ** 2) / 300)
        glucose = np.clip(glucose, 0.3, 8.0)

        # DO: affected by kLa (scale-dependent oxygen transfer)
        # Lower kLa → more DO drop per cell density
        kla_effect = base_kla / 80.0  # Normalize to bench scale
        do_response = 45 + 10 * np.sin(0.02 * time) - 25 * (cell_density / max_density) / kla_effect

        # Add mixing time effect: larger scale = more DO fluctuation
        mixing_noise = np.random.normal(0, 0.5 + 1.5 * (mixing_time / 10), n_samples)
        do_measured = do_response + mixing_noise
        do_measured = np.clip(do_measured, 15, 60)

        # CO2: proportional to metabolic activity (scale-independent biology)
        co2 = 2.5 + 3.5 * (cell_density / max_density) * (1 - glucose / 8.0)
        co2 += np.random.normal(0, 0.3, n_samples)
        co2 = np.clip(co2, 1.0, 7.0)

        # Base addition: responds to lactate from glucose metabolism
        base_rate = 1.5 + 10 * (1 - glucose / 8.0) ** 2 * (cell_density / max_density)
        base_rate += np.random.normal(0, 0.6, n_samples)
        base_rate = np.clip(base_rate, 0, 18)

        # Temperature: thermal lag increases with scale
        # Larger vessels: slower response, more drift
        temp_setpoint = 37.0
        temp_noise = np.random.normal(0, 0.1 + 0.15 * (scale_volume / 10000), n_samples)
        temp_measured = temp_setpoint + temp_noise
        temp_measured = np.clip(temp_measured, 36.5, 37.8)

        # Scale-specific systematic offset in glucose (the key challenge)
        # Due to heterogeneity, production-scale glucose-to-DO relationship shifts
        scale_offset = -0.3 * (scale_volume / 10000)  # Production reads slightly lower
        glucose_measured = glucose + scale_offset
        glucose_measured = np.clip(glucose_measured, 0.2, 8.5)

        batch_df = pd.DataFrame({
            'batch_id': batch_id,
            'scale': scale_name,
            'volume_L': scale_volume,
            'time_h': time,
            'DO_%': do_measured,
            'CO2_%': co2,
            'base_rate_mL_h': base_rate,
            'temp_C': temp_measured,
            'glucose_g_L': glucose_measured,
            'cell_density_e6': cell_density / 1e6
        })

        all_batches.append(batch_df)

    return pd.concat(all_batches, ignore_index=True)


# Generate multi-scale datasets
print("Generating multi-scale bioprocess data...")
print("=" * 70)

# Bench scale: abundant data
bench_data = generate_batch_data(
    n_batches=50,
    scale_name='Bench',
    scale_volume=3,
    base_kla=80,  # High oxygen transfer
    mixing_time=3  # Seconds
)
print(f"✓ Bench (3L): {bench_data['batch_id'].nunique()} batches, {len(bench_data)} samples")

# Pilot scale: moderate data
pilot_data = generate_batch_data(
    n_batches=12,
    scale_name='Pilot',
    scale_volume=200,
    base_kla=45,  # Medium oxygen transfer
    mixing_time=20  # Seconds
)
print(f"✓ Pilot (200L): {pilot_data['batch_id'].nunique()} batches, {len(pilot_data)} samples")

# Production scale: limited data (the real challenge)
production_data = generate_batch_data(
    n_batches=5,  # Only 5 successful batches!
    scale_name='Production',
    scale_volume=10000,
    base_kla=30,  # Low oxygen transfer (scale limitation)
    mixing_time=50  # Seconds
)
print(f"✓ Production (10,000L): {production_data['batch_id'].nunique()} batches, {len(production_data)} samples")
print("=" * 70)


# ============================================================================
# PART 2: Define Neural Network Architecture for Transfer Learning
# ============================================================================

class BioprocessTransferModel(nn.Module):
    """
    Neural network designed for transfer learning.

    Architecture:
    - Feature extractor (early layers): learns general bioprocess dynamics
    - Scale adapter (middle layers): adapts to scale-specific physics
    - Predictor (final layers): maps to glucose output

    For transfer learning:
    - Pre-train entire network on bench data
    - Freeze feature extractor
    - Fine-tune adapter + predictor on production data
    """

    def __init__(self, input_dim=4):
        super(BioprocessTransferModel, self).__init__()

        # Feature extractor: generic temporal patterns
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # Scale adapter: learns scale-specific mappings
        self.scale_adapter = nn.Sequential(
            nn.Linear(32, 24),
            nn.ReLU(),
            nn.Dropout(0.15)
        )

        # Predictor: final glucose prediction
        self.predictor = nn.Sequential(
            nn.Linear(24, 12),
            nn.ReLU(),
            nn.Linear(12, 1)
        )

    def forward(self, x):
        features = self.feature_extractor(x)
        adapted = self.scale_adapter(features)
        output = self.predictor(adapted)
        return output

    def freeze_feature_extractor(self):
        """Freeze early layers for transfer learning"""
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

    def unfreeze_all(self):
        """Unfreeze all layers"""
        for param in self.parameters():
            param.requires_grad = True


# ============================================================================
# PART 3: Training Functions
# ============================================================================

def prepare_data(df, feature_cols, target_col):
    """Prepare features and target"""
    X = df[feature_cols].values
    y = df[target_col].values.reshape(-1, 1)
    return X, y


def train_model(model, train_loader, val_loader, n_epochs, lr=0.001, verbose=True):
    """Standard training loop"""
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    train_losses = []
    val_losses = []

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

        if verbose and (epoch + 1) % 25 == 0:
            print(f"  Epoch {epoch + 1}/{n_epochs}, Train: {train_loss:.4f}, Val: {val_loss:.4f}")

    return train_losses, val_losses


def evaluate_model(model, X, y, scaler_X, scaler_y):
    """Evaluate model and return predictions + metrics"""
    X_scaled = scaler_X.transform(X)
    model.eval()
    with torch.no_grad():
        y_pred_scaled = model(torch.FloatTensor(X_scaled)).numpy()

    y_pred = scaler_y.inverse_transform(y_pred_scaled).flatten()
    y_true = y.flatten()

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    return y_pred, rmse, r2


# ============================================================================
# PART 4: Experimental Scenarios
# ============================================================================

feature_cols = ['DO_%', 'CO2_%', 'base_rate_mL_h', 'temp_C']
target_col = 'glucose_g_L'

# Split production data: 3 for training, 2 for testing
production_train_batches = [0, 1, 2]
production_test_batches = [3, 4]

prod_train = production_data[production_data['batch_id'].isin(production_train_batches)]
prod_test = production_data[production_data['batch_id'].isin(production_test_batches)]

print("\n" + "=" * 70)
print("SCENARIO COMPARISON: PRODUCTION-SCALE GLUCOSE PREDICTION")
print("=" * 70)

# ============================================================================
# SCENARIO 1: Train from scratch on limited production data (BASELINE)
# ============================================================================

print("\n[SCENARIO 1] Training from scratch on 3 production batches...")

X_prod_train, y_prod_train = prepare_data(prod_train, feature_cols, target_col)
X_prod_test, y_prod_test = prepare_data(prod_test, feature_cols, target_col)

# Scale data
scaler_X_prod = StandardScaler()
scaler_y_prod = StandardScaler()

X_prod_train_scaled = scaler_X_prod.fit_transform(X_prod_train)
y_prod_train_scaled = scaler_y_prod.fit_transform(y_prod_train)

X_prod_test_scaled = scaler_X_prod.transform(X_prod_test)
y_prod_test_scaled = scaler_y_prod.transform(y_prod_test)

# Create data loaders
train_dataset_scratch = TensorDataset(
    torch.FloatTensor(X_prod_train_scaled),
    torch.FloatTensor(y_prod_train_scaled)
)
train_loader_scratch = DataLoader(train_dataset_scratch, batch_size=32, shuffle=True)

val_dataset_scratch = TensorDataset(
    torch.FloatTensor(X_prod_test_scaled),
    torch.FloatTensor(y_prod_test_scaled)
)
val_loader_scratch = DataLoader(val_dataset_scratch, batch_size=32, shuffle=False)

# Train model from scratch
model_scratch = BioprocessTransferModel(input_dim=4)
losses_scratch_train, losses_scratch_val = train_model(
    model_scratch, train_loader_scratch, val_loader_scratch,
    n_epochs=150, lr=0.001, verbose=False
)

# Evaluate
y_pred_scratch, rmse_scratch, r2_scratch = evaluate_model(
    model_scratch, X_prod_test, y_prod_test, scaler_X_prod, scaler_y_prod
)

print(f"✗ From-scratch model: RMSE = {rmse_scratch:.3f} g/L, R² = {r2_scratch:.3f}")
print(f"  Problem: Severe overfitting on {len(production_train_batches)} batches")

# ============================================================================
# SCENARIO 2: Naive bench model applied to production (WORST)
# ============================================================================

print("\n[SCENARIO 2] Naive bench-scale model (no adaptation)...")

X_bench, y_bench = prepare_data(bench_data, feature_cols, target_col)

# Split bench data
split_idx = int(0.85 * len(X_bench))
X_bench_train, y_bench_train = X_bench[:split_idx], y_bench[:split_idx]
X_bench_val, y_bench_val = X_bench[split_idx:], y_bench[split_idx:]

# Scale bench data
scaler_X_bench = StandardScaler()
scaler_y_bench = StandardScaler()

X_bench_train_scaled = scaler_X_bench.fit_transform(X_bench_train)
y_bench_train_scaled = scaler_y_bench.fit_transform(y_bench_train)

X_bench_val_scaled = scaler_X_bench.transform(X_bench_val)
y_bench_val_scaled = scaler_y_bench.transform(y_bench_val)

# Train on bench
train_dataset_bench = TensorDataset(
    torch.FloatTensor(X_bench_train_scaled),
    torch.FloatTensor(y_bench_train_scaled)
)
train_loader_bench = DataLoader(train_dataset_bench, batch_size=64, shuffle=True)

val_dataset_bench = TensorDataset(
    torch.FloatTensor(X_bench_val_scaled),
    torch.FloatTensor(y_bench_val_scaled)
)
val_loader_bench = DataLoader(val_dataset_bench, batch_size=64, shuffle=False)

model_bench = BioprocessTransferModel(input_dim=4)
losses_bench_train, losses_bench_val = train_model(
    model_bench, train_loader_bench, val_loader_bench,
    n_epochs=100, lr=0.001, verbose=False
)

# Apply to production (wrong scalers!)
y_pred_naive, rmse_naive, r2_naive = evaluate_model(
    model_bench, X_prod_test, y_prod_test, scaler_X_bench, scaler_y_bench
)

print(f"✗ Naive bench model: RMSE = {rmse_naive:.3f} g/L, R² = {r2_naive:.3f}")
print(f"  Problem: Scale-dependent physics ignored, systematic bias")

# ============================================================================
# SCENARIO 3: Transfer learning (BEST)
# ============================================================================

print("\n[SCENARIO 3] Transfer learning: bench → production...")

# Step 1: Pre-train on bench data
print("  Step 1: Pre-training on 50 bench-scale batches...")
model_transfer = BioprocessTransferModel(input_dim=4)
losses_transfer_pretrain_train, losses_transfer_pretrain_val = train_model(
    model_transfer, train_loader_bench, val_loader_bench,
    n_epochs=100, lr=0.001, verbose=False
)

# Step 2: Freeze feature extractor
print("  Step 2: Freezing feature extractor...")
model_transfer.freeze_feature_extractor()

# Step 3: Fine-tune on production data with production scalers
print("  Step 3: Fine-tuning on 3 production batches...")

train_dataset_transfer = TensorDataset(
    torch.FloatTensor(X_prod_train_scaled),
    torch.FloatTensor(y_prod_train_scaled)
)
train_loader_transfer = DataLoader(train_dataset_transfer, batch_size=32, shuffle=True)

val_dataset_transfer = TensorDataset(
    torch.FloatTensor(X_prod_test_scaled),
    torch.FloatTensor(y_prod_test_scaled)
)
val_loader_transfer = DataLoader(val_dataset_transfer, batch_size=32, shuffle=False)

losses_transfer_finetune_train, losses_transfer_finetune_val = train_model(
    model_transfer, train_loader_transfer, val_loader_transfer,
    n_epochs=75, lr=0.0005, verbose=False  # Lower LR for fine-tuning
)

# Evaluate transfer model
y_pred_transfer, rmse_transfer, r2_transfer = evaluate_model(
    model_transfer, X_prod_test, y_prod_test, scaler_X_prod, scaler_y_prod
)

print(f"✓ Transfer learning: RMSE = {rmse_transfer:.3f} g/L, R² = {r2_transfer:.3f}")
print(f"  Improvement over scratch: {(1 - rmse_transfer / rmse_scratch) * 100:.1f}%")
print(f"  Improvement over naive: {(1 - rmse_transfer / rmse_naive) * 100:.1f}%")

# ============================================================================
# PART 5: Comprehensive Visualization
# ============================================================================

print("\n" + "=" * 70)
print("Generating comprehensive visualizations...")

fig = plt.figure(figsize=(20, 14))

# 1. Multi-scale data distribution
ax1 = plt.subplot(3, 4, 1)
for scale, color in [('Bench', 'blue'), ('Pilot', 'orange'), ('Production', 'red')]:
    # CORRECTED LINE:
    data = [bench_data, pilot_data, production_data][('Bench', 'Pilot', 'Production').index(scale)]
    ax1.scatter(data['DO_%'], data['glucose_g_L'], alpha=0.3, s=10, label=scale, color=color)
ax1.set_xlabel('DO (%)', fontsize=11, fontweight='bold')
ax1.set_ylabel('Glucose (g/L)', fontsize=11, fontweight='bold')
ax1.set_title('A. Scale-Dependent Data Distribution', fontsize=12, fontweight='bold', pad=10)
ax1.legend(frameon=True, shadow=True)
ax1.grid(True, alpha=0.3)

# 2. Training data availability
ax2 = plt.subplot(3, 4, 2)
scales = ['Bench\n(3L)', 'Pilot\n(200L)', 'Production\n(10kL)']
n_batches = [50, 12, 3]
colors_bar = ['blue', 'orange', 'red']
bars = ax2.bar(scales, n_batches, color=colors_bar, alpha=0.7, edgecolor='black', linewidth=2)
for bar, n in zip(bars, n_batches):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width() / 2., height,
             f'{n} batches\n({n * 400} samples)',
             ha='center', va='bottom', fontsize=10, fontweight='bold')
ax2.set_ylabel('Training Batches Available', fontsize=11, fontweight='bold')
ax2.set_title('B. The Small-N Problem at Scale', fontsize=12, fontweight='bold', pad=10)
ax2.set_ylim([0, 60])
ax2.grid(True, alpha=0.3, axis='y')

# 3. Learning curves: scratch vs transfer
ax3 = plt.subplot(3, 4, 3)
ax3.plot(losses_scratch_train, label='Scratch (train)', color='red', linewidth=2, alpha=0.7)
ax3.plot(losses_scratch_val, label='Scratch (val)', color='red', linewidth=2, linestyle='--', alpha=0.7)
ax3.plot(losses_transfer_finetune_train, label='Transfer (train)', color='green', linewidth=2, alpha=0.7)
ax3.plot(losses_transfer_finetune_val, label='Transfer (val)', color='green', linewidth=2, linestyle='--', alpha=0.7)
ax3.set_xlabel('Epoch (fine-tuning)', fontsize=11, fontweight='bold')
ax3.set_ylabel('MSE Loss', fontsize=11, fontweight='bold')
ax3.set_title('C. Learning Curves: Scratch vs Transfer', fontsize=12, fontweight='bold', pad=10)
ax3.legend(frameon=True, shadow=True, fontsize=9)
ax3.grid(True, alpha=0.3)
ax3.set_yscale('log')

# 4. Performance comparison bar chart
ax4 = plt.subplot(3, 4, 4)
methods = ['Scratch\n(3 batches)', 'Naive\nBench', 'Transfer\nLearning']
rmse_values = [rmse_scratch, rmse_naive, rmse_transfer]
colors_methods = ['red', 'orange', 'green']
bars = ax4.bar(methods, rmse_values, color=colors_methods, alpha=0.7, edgecolor='black', linewidth=2)
for bar, rmse in zip(bars, rmse_values):
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width() / 2., height,
             f'{rmse:.3f}',
             ha='center', va='bottom', fontsize=11, fontweight='bold')
ax4.set_ylabel('Test RMSE (g/L)', fontsize=11, fontweight='bold')
ax4.set_title('D. Performance Comparison', fontsize=12, fontweight='bold', pad=10)
ax4.set_ylim([0, max(rmse_values) * 1.3])
ax4.grid(True, alpha=0.3, axis='y')

# Add improvement annotation
improvement = (1 - rmse_transfer / rmse_scratch) * 100
ax4.text(0.5, 0.95, f'Transfer learning:\n{improvement:.1f}% better than scratch',
         transform=ax4.transAxes, fontsize=10, verticalalignment='top',
         horizontalalignment='center', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8),
         fontweight='bold')

# 5-7. Prediction vs actual for each scenario (test batches)
test_batch_4 = prod_test[prod_test['batch_id'] == 3]
test_batch_5 = prod_test[prod_test['batch_id'] == 4]

for i, (batch_df, batch_name) in enumerate([(test_batch_4, 'Batch 4'), (test_batch_5, 'Batch 5')]):
    if i == 0:
        continue  # Skip, will use combined view

ax5 = plt.subplot(3, 4, 5)
time_test = prod_test['time_h'].values
y_true_test = prod_test['glucose_g_L'].values

idx_4 = prod_test['batch_id'] == 3
idx_5 = prod_test['batch_id'] == 4

ax5.plot(time_test[idx_4], y_true_test[idx_4], 'k-', linewidth=3, alpha=0.8, label='True (Batch 4)')
ax5.plot(time_test[idx_5], y_true_test[idx_5], 'k--', linewidth=3, alpha=0.8, label='True (Batch 5)')
ax5.set_xlabel('Time (hours)', fontsize=11, fontweight='bold')
ax5.set_ylabel('Glucose (g/L)', fontsize=11, fontweight='bold')
ax5.set_title('E. Ground Truth: Test Batches', fontsize=12, fontweight='bold', pad=10)
ax5.legend(frameon=True, shadow=True, fontsize=9)
ax5.grid(True, alpha=0.3)

# 6. From-scratch predictions
ax6 = plt.subplot(3, 4, 6)
ax6.plot(time_test[idx_4], y_true_test[idx_4], 'k-', linewidth=2.5, alpha=0.6)
ax6.plot(time_test[idx_5], y_true_test[idx_5], 'k--', linewidth=2.5, alpha=0.6)
ax6.plot(time_test[idx_4], y_pred_scratch[idx_4], 'r-', linewidth=2, alpha=0.8, label='Scratch (Batch 4)')
ax6.plot(time_test[idx_5], y_pred_scratch[idx_5], 'r--', linewidth=2, alpha=0.8, label='Scratch (Batch 5)')
ax6.set_xlabel('Time (hours)', fontsize=11, fontweight='bold')
ax6.set_ylabel('Glucose (g/L)', fontsize=11, fontweight='bold')
ax6.set_title(f'F. From Scratch (RMSE: {rmse_scratch:.3f} g/L)', fontsize=12, fontweight='bold', pad=10)
ax6.legend(frameon=True, shadow=True, fontsize=9)
ax6.grid(True, alpha=0.3)

# 7. Naive bench model predictions
ax7 = plt.subplot(3, 4, 7)
ax7.plot(time_test[idx_4], y_true_test[idx_4], 'k-', linewidth=2.5, alpha=0.6)
ax7.plot(time_test[idx_5], y_true_test[idx_5], 'k--', linewidth=2.5, alpha=0.6)
ax7.plot(time_test[idx_4], y_pred_naive[idx_4], color='orange', linestyle='-', linewidth=2, alpha=0.8,
         label='Naive (Batch 4)')
ax7.plot(time_test[idx_5], y_pred_naive[idx_5], color='orange', linestyle='--', linewidth=2, alpha=0.8,
         label='Naive (Batch 5)')
ax7.set_xlabel('Time (hours)', fontsize=11, fontweight='bold')
ax7.set_ylabel('Glucose (g/L)', fontsize=11, fontweight='bold')
ax7.set_title(f'G. Naive Bench Model (RMSE: {rmse_naive:.3f} g/L)', fontsize=12, fontweight='bold', pad=10)
ax7.legend(frameon=True, shadow=True, fontsize=9)
ax7.grid(True, alpha=0.3)

# 8. Transfer learning predictions
ax8 = plt.subplot(3, 4, 8)
ax8.plot(time_test[idx_4], y_true_test[idx_4], 'k-', linewidth=2.5, alpha=0.8, label='True (Batch 4)')
ax8.plot(time_test[idx_5], y_true_test[idx_5], 'k--', linewidth=2.5, alpha=0.8, label='True (Batch 5)')
ax8.plot(time_test[idx_4], y_pred_transfer[idx_4], 'g-', linewidth=2, alpha=0.8, label='Transfer (Batch 4)')
ax8.plot(time_test[idx_5], y_pred_transfer[idx_5], 'g--', linewidth=2, alpha=0.8, label='Transfer (Batch 5)')
ax8.set_xlabel('Time (hours)', fontsize=11, fontweight='bold')
ax8.set_ylabel('Glucose (g/L)', fontsize=11, fontweight='bold')
ax8.set_title(f'H. Transfer Learning (RMSE: {rmse_transfer:.3f} g/L)', fontsize=12, fontweight='bold', pad=10)
ax8.legend(frameon=True, shadow=True, fontsize=9)
ax8.grid(True, alpha=0.3)

# 9. Parity plots comparison
ax9 = plt.subplot(3, 4, 9)
ax9.scatter(y_true_test, y_pred_scratch, alpha=0.5, s=30, color='red',
            edgecolors='black', linewidth=0.5, label=f'Scratch (R²={r2_scratch:.3f})')
ax9.scatter(y_true_test, y_pred_transfer, alpha=0.5, s=30, color='green',
            edgecolors='black', linewidth=0.5, label=f'Transfer (R²={r2_transfer:.3f})')
min_val = min(y_true_test.min(), y_pred_scratch.min(), y_pred_transfer.min())
max_val = max(y_true_test.max(), y_pred_scratch.max(), y_pred_transfer.max())
ax9.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, alpha=0.7)
ax9.set_xlabel('True Glucose (g/L)', fontsize=11, fontweight='bold')
ax9.set_ylabel('Predicted Glucose (g/L)', fontsize=11, fontweight='bold')
ax9.set_title('I. Parity Plot: Scratch vs Transfer', fontsize=12, fontweight='bold', pad=10)
ax9.legend(frameon=True, shadow=True, fontsize=9)
ax9.grid(True, alpha=0.3)
ax9.set_aspect('equal', adjustable='box')

# 10. Error distributions
ax10 = plt.subplot(3, 4, 10)
error_scratch = y_true_test - y_pred_scratch
error_transfer = y_true_test - y_pred_transfer
ax10.hist(error_scratch, bins=20, alpha=0.6, color='red', edgecolor='black',
          label=f'Scratch (σ={np.std(error_scratch):.3f})', density=True)
ax10.hist(error_transfer, bins=20, alpha=0.6, color='green', edgecolor='black',
          label=f'Transfer (σ={np.std(error_transfer):.3f})', density=True)
ax10.axvline(x=0, color='black', linestyle='--', linewidth=2)
ax10.set_xlabel('Prediction Error (g/L)', fontsize=11, fontweight='bold')
ax10.set_ylabel('Density', fontsize=11, fontweight='bold')
ax10.set_title('J. Error Distribution Comparison', fontsize=12, fontweight='bold', pad=10)
ax10.legend(frameon=True, shadow=True, fontsize=9)
ax10.grid(True, alpha=0.3, axis='y')

# 11. Residuals over time
ax11 = plt.subplot(3, 4, 11)
ax11.scatter(time_test, error_scratch, alpha=0.5, s=20, color='red', label='Scratch')
ax11.scatter(time_test, error_transfer, alpha=0.5, s=20, color='green', label='Transfer')
ax11.axhline(y=0, color='black', linestyle='--', linewidth=2)
ax11.set_xlabel('Time (hours)', fontsize=11, fontweight='bold')
ax11.set_ylabel('Residuals (g/L)', fontsize=11, fontweight='bold')
ax11.set_title('K. Residuals Over Time', fontsize=12, fontweight='bold', pad=10)
ax11.legend(frameon=True, shadow=True, fontsize=9)
ax11.grid(True, alpha=0.3)

# 12. Transfer learning architecture visualization
ax12 = plt.subplot(3, 4, 12)
ax12.axis('off')

# Draw architecture
architecture_text = """
TRANSFER LEARNING ARCHITECTURE

Step 1: Pre-train on Bench (50 batches)
┌─────────────────────────────────┐
│   Feature Extractor (64→32)     │ ← Generic patterns
│   [ReLU, BatchNorm, Dropout]    │
├─────────────────────────────────┤
│   Scale Adapter (32→24)         │ ← Scale-specific
│   [ReLU, Dropout]               │
├─────────────────────────────────┤
│   Predictor (24→12→1)           │ ← Glucose output
│   [ReLU, Linear]                │
└─────────────────────────────────┘

Step 2: Freeze Feature Extractor
┌═════════════════════════════════┐
║   Feature Extractor (FROZEN)    ║ 🔒
╠─────────────────────────────────╣
│   Scale Adapter (TRAINABLE)    │ 🔓
├─────────────────────────────────┤
│   Predictor (TRAINABLE)         │ 🔓
└─────────────────────────────────┘

Step 3: Fine-tune on Production (3 batches)
→ Adapt to scale-specific physics
→ Maintain learned temporal patterns
→ Avoid overfitting with frozen layers

Result: {:.1f}% improvement over scratch
""".format(improvement)

ax12.text(0.05, 0.95, architecture_text,
          transform=ax12.transAxes, fontsize=9, verticalalignment='top',
          horizontalalignment='left', family='monospace',
          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

plt.tight_layout()
plt.savefig('transfer_learning_scale_up.png', dpi=300, bbox_inches='tight')
print("Figure saved as 'transfer_learning_scale_up.png'")
plt.show()

# ============================================================================
# PART 6: Summary Statistics
# ============================================================================

print("\n" + "=" * 70)
print("TRANSFER LEARNING SCALE-UP ANALYSIS: SUMMARY")
print("=" * 70)
print("\nDATA LANDSCAPE:")
print(f"  Bench scale (3L):        {bench_data['batch_id'].nunique()} batches, {len(bench_data):,} samples")
print(f"  Pilot scale (200L):      {pilot_data['batch_id'].nunique()} batches, {len(pilot_data):,} samples")
print(f"  Production (10,000L):    {production_data['batch_id'].nunique()} batches (3 train, 2 test)")
print("\nMODEL PERFORMANCE (Production Test Set):")
print(f"  {'Method':<25} {'RMSE (g/L)':<15} {'R²':<10} {'Status'}")
print(f"  {'-' * 65}")
print(f"  {'From Scratch (3 batches)':<25} {rmse_scratch:<15.3f} {r2_scratch:<10.3f} {'❌ Overfitting'}")
print(f"  {'Naive Bench Model':<25} {rmse_naive:<15.3f} {r2_naive:<10.3f} {'❌ Domain mismatch'}")
print(f"  {'Transfer Learning':<25} {rmse_transfer:<15.3f} {r2_transfer:<10.3f} {'✅ Best'}")
print("\nIMPROVEMENTS:")
print(f"  Transfer vs. Scratch:     {(1 - rmse_transfer / rmse_scratch) * 100:>6.1f}% error reduction")
print(f"  Transfer vs. Naive:       {(1 - rmse_transfer / rmse_naive) * 100:>6.1f}% error reduction")
print("\nKEY INSIGHTS:")
print("  • Small-N problem is fundamental to bioprocess scale-up")
print("  • Naive model transfer fails due to scale-dependent physics")
print("  • From-scratch training overfits with limited production data")
print("  • Transfer learning leverages abundant bench data while adapting")
print("  • Frozen feature extraction preserves learned temporal patterns")
print("=" * 70)

print("\n" + "=" * 70)
print("PRACTICAL DEPLOYMENT STRATEGY")
print("=" * 70)
print("""
1. PRE-TRAINING PHASE (Offline, once)
   • Compile all historical bench-scale batches (target: 30+ batches)
   • Train feature extractor on generic bioprocess dynamics
   • Validate on held-out bench batches
   • Save model checkpoint

2. SCALE-UP PREPARATION
   • Run 2-3 qualification batches at target scale
   • Collect same sensor suite as bench (DO, CO2, base, temp)
   • Obtain reference glucose measurements (offline)

3. FINE-TUNING (1-2 days)
   • Load pre-trained model
   • Freeze feature extractor layers
   • Fine-tune adapter + predictor on scale-up batches
   • Use lower learning rate (0.0005 vs 0.001)
   • Validate on hold-out batch if available

4. DEPLOYMENT & MONITORING
   • Deploy with uncertainty estimation
   • Collect reference measurements periodically
   • Trigger online adaptation if drift detected (see Post #2)
   • Log all predictions for continuous improvement

5. CONTINUOUS IMPROVEMENT
   • Add new production batches to fine-tuning set
   • Periodically retrain adapter (monthly or per campaign)
   • Consider pilot-scale intermediate transfer step
   • Document performance for regulatory validation
""")
print("=" * 70)