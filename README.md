Soft Sensor for Bioreactor (Demo)
=================================

This repository contains a minimal, self‑contained demo of a soft sensor that estimates glucose concentration in a fed‑batch bioreactor using routinely available online signals. It simulates synthetic data and trains a small neural network to predict glucose from:

- Time (hours)
- Dissolved Oxygen (DO, %)
- CO₂ (%)
- Base addition rate (mL/h)

![Bioreactor soft sensor results: training history, parity (R²), residuals, time series, sensitivity, error distribution.](bioreactor_soft_sensor.png)

Key features
------------
- Synthetic but realistic process data generation
- Train/val/test split with scaling
- Lightweight fully connected neural network (PyTorch)
- Clear visualizations: training history, parity, residuals, time series, sensitivity, error distribution
- CLI with sensible defaults and options

Quickstart
----------

1. Create a virtual environment (recommended). If you use `uv` (fast Python package manager):

   - Install uv (if needed): https://docs.astral.sh/uv/
   - Create and sync the environment based on `pyproject.toml`:

     uv sync

2. Install PyTorch (required for training). Follow the official selector for your platform and CUDA needs:

   https://pytorch.org/get-started/locally/

   Example (CPU only):

     pip install torch --index-url https://download.pytorch.org/whl/cpu

3. Run the demo script:

   - Show CLI help:

     uv run python app.py -h

   - Run training with defaults and save the figure:

     uv run python app.py --epochs 150 --n-samples 500 --no-show --out bioreactor_soft_sensor.png

Repository structure
--------------------

- app.py — main script/module with:
  - Data generation (generate_bioreactor_data)
  - Data preparation and scalers
  - Model training loop
  - Visualization and reporting
  - CLI entry point guarded by `if __name__ == "__main__":`
- pyproject.toml — project metadata and core dependencies
- uv.lock — lock file for uv (if used)
- README.md — this document
- LICENSE — MIT license text

Requirements
------------

- Python 3.9+
- NumPy, Pandas, scikit‑learn, Matplotlib (declared in pyproject)
- PyTorch (not pinned here; install per your platform)

Notes on reproducibility and performance
---------------------------------------

- The script sets NumPy (and torch, if available) random seeds for basic reproducibility.
- The dataset is synthetic; absolute numbers are illustrative, not validated against real process data.
- Training is lightweight and intended for demonstration; tune architecture and hyperparameters for real use cases.

License
-------

This project is licensed under the MIT License.

You should have received a copy of the MIT License along with this project as the file `LICENSE`. If not, see https://opensource.org/licenses/MIT.

Citation
--------

If you use this demo as a starting point, please cite this repository in your work.
