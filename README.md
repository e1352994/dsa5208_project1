# Distributed SGD on NYC Taxi Dataset (2022)

This project applies distributed stochastic gradient descent (SGD) using MPI (`mpi4py`) to train a simple neural network on the 2022 NYC Taxi dataset. The training is parallelized across multiple processes using MPI.

## Project Structure

```
.
├── nytaxi2022_data_clean.ipynb     # Data preprocessing steps
├── nytaxi2022_cleaned.csv          # Cleaned dataset (generated)
├── nytaxi2022_sgd.py               # Distributed SGD training script
├── sgd_training_results.csv        # Final RMSEs and time (generated)
├── sgd_training_losses.csv         # Loss history per epoch (generated)
├── plot_training_results.py        # (To be added) Plotting script for training curves and summary

```

## Software Requirements
- Python 3.8+
- MPI installed
- Python packages:
```
pip install pandas numpy mpi4py scikit-learn
```


## How to Run

### 1. Data Cleaning

Open and run the Jupyter notebook:

```
jupyter notebook nytaxi2022_data_clean.ipynb
```

This will:
- Load and clean the raw NYC Taxi data
- Handle missing values, outliers, and normalize features
- Save nytaxi2022_cleaned.csv as the processed dataset

### 2. Distributed Training

Run the training script using mpirun:

```
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
You can optionally run a plotting script (to be added) to visualize:
- RMSE vs Batch Size
- Training Loss Curves per Configuration
- Time vs Accuracy tradeoffs

```
python plot_training_results.py
```
Plotted results will be saved in the root folder



