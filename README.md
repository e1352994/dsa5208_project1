# Distributed SGD on NYC Taxi Dataset (2022)

This project applies distributed stochastic gradient descent (SGD) using MPI (`mpi4py`) to train a simple neural network on the 2022 NYC Taxi dataset. The training is parallelized across multiple processes using MPI.

## Project Structure

```bash
.
├── nytaxi2022_data_clean.ipynb         # Data preprocessing steps
├── nytaxi2022_cleaned.csv              # Cleaned dataset (generated)
├── nytaxi2022_sgd_trial.py             # Trial run of distributed SGD
├── nytaxi2022_sgd_trial_plotting.ipynb # Visualizations for trial run
├── nytaxi2022_sgd.py                   # Final distributed SGD training script
├── nytaxi2022_result_plotting.ipynb    # Visualizations for final results
├── sgd_training_results_trial.csv      # Trial run results (generated)
├── sgd_training_losses_trial.csv       # Trial run loss history per epoch (generated)
├── sgd_training_results.csv            # Final RMSEs and time (generated)
└── sgd_training_losses.csv             # Loss history per epoch (generated)

```

## Software Requirements

- Python 3.8+
- MPI installed
- Python packages:

```bash
pip install pandas numpy mpi4py scikit-learn
```

## How to Run

### 1. Data Cleaning

The raw dataset file must be named **`nytaxi2022.csv`** and placed in the **project root directory**.  

To preprocess the raw data and generate the cleaned dataset, run:

```bash
jupyter notebook nytaxi2022_data_clean.ipynb
```

This will:

- Load and clean the raw NYC Taxi data
- Handle missing values, outliers, and normalize features
- Produce **`nytaxi2022_cleaned.csv`** as the processed dataset

⚠️ **Note:** Data cleaning may take time. If you want to skip this step, you can directly use the cleaned file. Ensure that **`nytaxi2022_cleaned.csv`** is already present in the project root folder before going to the next step

### 2. Distributed Training

Make sure the cleaned dataset **`nytaxi2022_cleaned.csv`** is in the **project root directory**.

Run the training script using mpirun:

```bash
mpirun -n <NUM_PROCESSES> python nytaxi2022_sgd.py
```

Please change <NUM_PROCESSES> to any number of processes you want to run in parallel based on your CPU cores

The script will:

- Load and distribute the cleaned dataset to all processes
- Train a simple 1-hidden-layer NN using distributed SGD
- Test various configurations (activation functions + batch sizes)

After training, the following files will be generated and saved **only on Rank 0**:

- sgd_training_results.csv (Contains final RMSE and training time for each configuration)
- sgd_training_losses.csv (Contains training loss history per epoch (one row per epoch))

### 3. Plotting Results (Optional)

You can optionally run the plotting script (nytaxi2022_result_plotting.ipynb) to visualize:

- RMSE vs Batch Size
- Training Loss Curves per Configuration
- Time vs Accuracy tradeoffs

```bash
jupyter notebook nytaxi2022_result_plotting.ipynb
```

Plotted results will be saved in the root folder
