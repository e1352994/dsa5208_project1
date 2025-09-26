from mpi4py import MPI
import pandas as pd
import numpy as np
import time
import os
import csv
import gc
from sklearn.model_selection import train_test_split

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()       # Process ID
size = comm.Get_size()       # Total number of processes

# Define tag constants for MPI communication
TAG_X_TRAIN = 11
TAG_Y_TRAIN = 12
TAG_X_TEST  = 13
TAG_Y_TEST  = 14

# -------------------- Data Loader & Distribution --------------------
def load_and_distribute_data(filename):
    X_train = y_train = X_test = y_test = None

    if rank == 0:
        # Load and split dataset
        print(f"[Rank 0] Loading data ...")
        df = pd.read_csv(filename)
        train_df, test_df = train_test_split(df, test_size=0.3, random_state=42)

        # Prepare training and test sets
        X_train_all = train_df.drop(columns=["total_amount"]).values
        y_train_all = train_df["total_amount"].values

        X_test_all = test_df.drop(columns=["total_amount"]).values
        y_test_all = test_df["total_amount"].values

        # Split training and test data into N chunks
        X_train_chunks = np.array_split(X_train_all, size)
        y_train_chunks = np.array_split(y_train_all, size)
        X_test_chunks = np.array_split(X_test_all, size)
        y_test_chunks = np.array_split(y_test_all, size)

        # Send chunks to other processes
        for i in range(1, size):
            comm.send(X_train_chunks[i], dest=i, tag=TAG_X_TRAIN)
            comm.send(y_train_chunks[i], dest=i, tag=TAG_Y_TRAIN)
            comm.send(X_test_chunks[i], dest=i, tag=TAG_X_TEST)
            comm.send(y_test_chunks[i], dest=i, tag=TAG_Y_TEST)
        
        # Keep data in rank 0        
        X_train = X_train_chunks[0]
        y_train = y_train_chunks[0]
        X_test = X_test_chunks[0]
        y_test = y_test_chunks[0]
        
        # Free memory
        del df, train_df, test_df, X_train_all, y_train_all, X_test_all, y_test_all
        del X_train_chunks, y_train_chunks, X_test_chunks, y_test_chunks
        gc.collect()

    else:
        # Receive data in other ranks
        X_train = comm.recv(source=0, tag=TAG_X_TRAIN)
        y_train = comm.recv(source=0, tag=TAG_Y_TRAIN)
        X_test = comm.recv(source=0, tag=TAG_X_TEST)
        y_test = comm.recv(source=0, tag=TAG_Y_TEST)
        
    print(f"[Rank {rank}] Received train: {X_train.shape}, test: {X_test.shape}")

    return X_train, y_train, X_test, y_test


# -------------------- Activation Functions --------------------
# ReLU
def relu(x): 
    return np.maximum(0, x)

def relu_deriv(x): 
    return (x > 0).astype(float)

# Tanh
def tanh(x): 
    return np.tanh(x)

def tanh_deriv(x): 
    return 1 - np.tanh(x)**2

# Sigmoid
def sigmoid(x): 
    return 1 / (1 + np.exp(-x))
def sigmoid_deriv(x):
    s = sigmoid(x)
    return s * (1 - s)

activation_functions = {
    "relu": (relu, relu_deriv),
    "tanh": (tanh, tanh_deriv),
    "sigmoid": (sigmoid, sigmoid_deriv)
}


# -------------------- RMSE --------------------
def compute_rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred)**2))


# -------------------- SGD Training --------------------
def run_distributed_sgd(X_train, y_train, X_test, y_test,
                        input_dim, hidden_dim,
                        activation_name, batch_size,
                        num_epochs, learning_rate):
    """
    Perform distributed stochastic gradient descent using MPI.

    Parameters:
        X_train, y_train: Training data for this process.
        X_test, y_test: Test set for this process.
        input_dim (int): Number of input features.
        hidden_dim (int): Number of hidden units.
        activation_name (str): Activation function name ("relu", "tanh", "sigmoid").
        batch_size (int): Mini-batch size.
        num_epochs (int): Max number of training epochs.
        learning_rate (float): Learning rate for SGD.
    """
    
    print(f"[Rank {rank}] Starting training with {activation_name}, batch size {batch_size}, training set {X_train.shape}, test set {X_test.shape} ...")
    
    # Get activation function
    activation, activation_deriv = activation_functions[activation_name]
    np.random.seed(42 + rank)

    # Initialize weights
    # input to hidden
    W1 = np.random.randn(input_dim, hidden_dim).astype(np.float64) * 0.01
    b1 = np.zeros((1, hidden_dim), dtype=np.float64)
    # hidden to output
    W2 = np.random.randn(hidden_dim, 1).astype(np.float64) * 0.01
    b2 = np.zeros((1, 1), dtype=np.float64)

    # To keep track of training loss and time
    train_loss_history = []
    start_time = time.time()

    # Convergence parameters
    epsilon = 1e-4         # Minimum change in loss
    patience = 5           # Stop if no improvement for this many epochs
    no_improve_count = 0   # How many epochs we've seen no improvement
    best_loss = float('inf')
    epoch = 0

    while epoch < num_epochs:
        # Shuffle all data at the start of each epoch
        perm = np.random.permutation(len(X_train))
        X_train_shuffled = X_train[perm]
        y_train_shuffled = y_train[perm]
        
        epoch_loss = 0
        num_batches = 0

        # Iterate through all mini-batches in the epoch
        for start_idx in range(0, len(X_train), batch_size):
            end_idx = min(start_idx + batch_size, len(X_train))
            X_batch = X_train_shuffled[start_idx:end_idx].astype(np.float64)
            y_batch = y_train_shuffled[start_idx:end_idx].reshape(-1, 1).astype(np.float64)

            # Forward
            Z1 = X_batch @ W1 + b1                      # linear output of hidden layer
            A1 = activation(Z1.astype(np.float64))      # activation output of hidden layer
            Z2 = A1 @ W2 + b2                           # final predicted value
            y_pred = Z2
            
            loss = np.mean((y_pred - y_batch)**2)
            epoch_loss += loss
            num_batches += 1
            
            # Backward
            dZ2 = (y_pred - y_batch) / X_batch.shape[0]
            dW2 = A1.T @ dZ2
            db2 = np.sum(dZ2, axis=0, keepdims=True)
            dA1 = dZ2 @ W2.T
            dZ1 = dA1 * activation_deriv(Z1)
            dW1 = X_batch.T @ dZ1
            db1 = np.sum(dZ1, axis=0, keepdims=True)

            # Update weights
            W1 -= learning_rate * dW1
            b1 -= learning_rate * db1
            W2 -= learning_rate * dW2
            b2 -= learning_rate * db2
            
        # Average loss for this epoch
        avg_epoch_loss = epoch_loss / num_batches
        train_loss_history.append(avg_epoch_loss)

        # Early stopping check
        if avg_epoch_loss < best_loss - epsilon:
            best_loss = avg_epoch_loss
            no_improve_count = 0
        else:
            no_improve_count += 1
            if no_improve_count >= patience:
                print(f"[Rank {rank}] Early stopping at epoch {epoch} with loss {avg_epoch_loss:.6f}")
                break
        
        if epoch % 5 == 0:
            print(f"[Rank {rank}] Epoch {epoch}, Loss: {loss:.6f}")  
        
        epoch += 1

    train_time = time.time() - start_time
    print(f"[Rank {rank}] Training complete in {train_time:.2f} sec over {epoch} epochs")
    
    # Average model parameters across all processes
    def average_weights(var):
        var_avg = np.zeros_like(var)
        comm.Allreduce(var, var_avg, op=MPI.SUM)
        return var_avg / size

    comm.Barrier()      # Ensure all processes have finished training
    W1 = average_weights(W1)
    b1 = average_weights(b1)
    W2 = average_weights(W2)
    b2 = average_weights(b2)

    # Evaluate on local training set
    print(f"[Rank {rank}] Evaluating on training set ...")
    Z1_train = X_train @ W1 + b1
    A1_train = activation(Z1_train.astype(np.float64))
    y_train_pred = A1_train @ W2 + b2
    local_rmse_train = compute_rmse(y_train, y_train_pred.squeeze())

    # Evaluate on local test set
    print(f"[Rank {rank}] Evaluating on test set ...")
    Z1_test = X_test @ W1 + b1
    A1_test = activation(Z1_test.astype(np.float64))
    y_test_pred = A1_test @ W2 + b2
    local_rmse_test = compute_rmse(y_test, y_test_pred.squeeze())

    # Reduce metrics
    print(f"[Rank {rank}] Reducing results ...")
    avg_train_rmse = comm.reduce(local_rmse_train, op=MPI.SUM, root=0)
    avg_test_rmse = comm.reduce(local_rmse_test, op=MPI.SUM, root=0)
    total_time = comm.reduce(train_time, op=MPI.SUM, root=0)

    if rank == 0:
        # Save results
        output_path = "sgd_results.csv"
        file_exists = os.path.isfile(output_path)

        with open(output_path, mode="a", newline="") as f:
            writer = csv.writer(f)
            
            # Write headers if file does not exist
            if not file_exists:
                writer.writerow(["num_of_process", "activation", "batch_size", "train_rmse", "test_rmse", "avg_time_sec"])
            
            # Append current run result
            writer.writerow([
                size,
                activation_name,
                batch_size,
                round(avg_train_rmse / size, 4),
                round(avg_test_rmse / size, 4),
                round(total_time / size, 2)
            ])
                
        print(f"\nActivation: {activation_name}, Batch Size: {batch_size}")
        print(f"   Train RMSE: {avg_train_rmse / size:.4f}")
        print(f"   Test  RMSE: {avg_test_rmse / size:.4f}")
        print(f"   Avg Time : {total_time / size:.2f} sec\n")
        
    # Free memory
    del W1, b1, W2, b2, Z1, A1, Z2, y_pred
    del Z1_train, A1_train, y_train_pred, Z1_test, A1_test, y_test_pred
    gc.collect()
        
    return train_loss_history

# -------------------- Save Loss History --------------------
def save_loss_history(loss_history, activation, batch_size):
    out_dir = "loss_logs"
    os.makedirs(out_dir, exist_ok=True)
    
    # Build filename and path
    filename = f"{size}_process_loss_{activation}_bs{batch_size}.csv"
    filepath = os.path.join(out_dir, filename)
    
    df = pd.DataFrame(loss_history, columns=["train_loss"])
    df["epoch"] = np.arange(len(loss_history))
    df["rank"] = rank
    
    # Gather all losses at root
    all_losses = comm.gather(df, root=0)
    
    comm.Barrier()
    
    if rank == 0:
        full_df = pd.concat(all_losses, ignore_index=True)
        if os.path.exists(filepath):
            os.remove(filepath)
        full_df.to_csv(filepath, index=False)
        
        print(f"[Rank 0] Saved combined loss history to: {filepath}")
    else:
        print(f"[Rank {rank}] sent loss history to root")


# Run traning for different configurations (activation functions and batch sizes)
if __name__ == "__main__":
    filename = "nytaxi2022_cleaned.csv"
    
    # Load and distribute data
    X_train, y_train, X_test, y_test = load_and_distribute_data(filename)
    
    # Training parameters
    num_epochs = 50
    learning_rate = 0.01
    input_dim = X_train.shape[1]
    hidden_dim = 32
    activations = ["relu", "tanh", "sigmoid"]
    batch_sizes = [512, 1024, 2048, 4096, 8192]
    

    # Train model using SGD with different activation functions and batch sizes
    for activation in activations:
        for batch_size in batch_sizes:
            if rank == 0:
                print(f"\n==== Strated training for activation {activation} with batch size {batch_size} ====")
            
            # Synchronize all processes before starting new run
            comm.Barrier()
            
            loss_history = run_distributed_sgd(
                X_train=X_train, y_train=y_train,
                X_test=X_test, y_test=y_test,
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                activation_name=activation,
                batch_size=batch_size,
                num_epochs=50,
                learning_rate=0.01
            )
            
            if loss_history is not None:
                save_loss_history(loss_history, activation, batch_size)
            
            # Synchronize all processes before next run
            comm.Barrier()
            
            if rank == 0:
                print(f"==== Completed training for activation {activation} with batch size {batch_size} ==== \n")
                
            # Free memory
            del loss_history
            gc.collect()
            
    comm.Barrier()  # Ensure all ranks reach this point
    if rank == 0:
        print("[Rank 0] All processes finished. Exiting now.")
    
    # Finalize MPI safely
    MPI.Finalize()