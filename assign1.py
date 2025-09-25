import numpy as np
import pandas as pd
from mpi4py import MPI

'''
mpiexec -n 4 python3 /Users/vald/Downloads/DSA5208分布式计算/assignment1/assign1.py
'''

# MPI 初始化
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# 超参数
input_dim = 100   # 特征维度
hidden_dim = 64
output_dim = 1
lr = 0.01
epochs = 10
batch_size = 128

# ===== 1. 数据加载与划分 =====
if rank == 0:
    # 用 pandas 读取真实数据
    data = pd.read_csv('/Users/vald/Downloads/DSA5208分布式计算/assignment1/nytaxi2022_cleaned.csv')
    # data.head()
    # 选择特征和目标
    features = [
        "trip_duration",
        "passenger_count",
        "trip_distance",
        "RatecodeID",
        "PULocationID",
        "DOLocationID",
        "payment_type",
        "extra"
    ]
    target = "total_amount"
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_all = scaler.fit_transform(data[features].values)
    y_all = data[target].values.reshape(-1, 1)
else:
    X_all, y_all = None, None

# 广播数据大小信息
if rank == 0:
    n_samples = X_all.shape[0]
    input_dim = X_all.shape[1]
else:
    n_samples = None
    input_dim = None
n_samples = comm.bcast(n_samples, root=0)
input_dim = comm.bcast(input_dim, root=0)
hidden_dim = 64
output_dim = 1

# 划分训练集和测试集
if rank == 0:
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.3, random_state=42)
else:
    X_train = X_test = y_train = y_test = None
# 广播训练集和测试集大小
if rank == 0:
    n_train = X_train.shape[0]
    n_test = X_test.shape[0]
else:
    n_train = n_test = None
n_train = comm.bcast(n_train, root=0)
n_test = comm.bcast(n_test, root=0)

# 每个进程切分数据
rows_per_proc = n_train // size
start = rank * rows_per_proc
end = (rank+1) * rows_per_proc if rank != size-1 else n_train
if rank == 0:
    local_X = X_train[start:end]
    local_y = y_train[start:end]
    for r in range(1, size):
        s = r * rows_per_proc
        e = (r+1) * rows_per_proc if r != size-1 else n_train
        comm.send(X_train[s:e], dest=r, tag=11)
        comm.send(y_train[s:e], dest=r, tag=12)
else:
    local_X = comm.recv(source=0, tag=11)
    local_y = comm.recv(source=0, tag=12)
n_local = local_X.shape[0]

# ===== 2. 模型初始化 =====
np.random.seed(42)
W1 = np.random.randn(input_dim, hidden_dim) * 0.01
b1 = np.zeros((1, hidden_dim))
W2 = np.random.randn(hidden_dim, output_dim) * 0.01
b2 = np.zeros((1, output_dim))

# ===== 3. 训练循环 =====
try:
    from tqdm import trange
except ImportError:
    trange = range
if rank == 0:
    epoch_iter = trange(epochs, desc="Epoch", ncols=80)
else:
    epoch_iter = range(epochs)
for epoch in epoch_iter:
    # 每个epoch前shuffle
    indices = np.random.permutation(n_local)
    local_X = local_X[indices]
    local_y = local_y[indices]
    for i in range(0, n_local, batch_size):
        batch_idx = np.random.choice(n_local, batch_size, replace=False) if n_local >= batch_size else np.arange(n_local)
        X_batch = local_X[batch_idx]
        y_batch = local_y[batch_idx]

        # 前向传播
        z1 = X_batch.dot(W1) + b1
        a1 = np.tanh(z1)
        z2 = a1.dot(W2) + b2
        y_pred = z2

        # 损失 (MSE)
        loss = np.mean((y_pred - y_batch) ** 2)

        # 反向传播
        dL_dy = 2 * (y_pred - y_batch) / batch_size
        dW2 = a1.T.dot(dL_dy)
        db2 = np.sum(dL_dy, axis=0, keepdims=True)
        da1 = dL_dy.dot(W2.T)
        dz1 = da1 * (1 - np.tanh(z1) ** 2)
        dW1 = X_batch.T.dot(dz1)
        db1 = np.sum(dz1, axis=0, keepdims=True)

        # ===== 4. 梯度同步 (Allreduce 平均化) =====
        for grad in [dW1, db1, dW2, db2]:
            comm.Allreduce(MPI.IN_PLACE, grad, op=MPI.SUM)
            grad /= size  # 求平均

        # 参数更新
        W1 -= lr * dW1
        b1 -= lr * db1
        W2 -= lr * dW2
        b2 -= lr * db2

    if rank == 0:
        print(f"Epoch {epoch+1}, Loss={loss:.4f}")
        # 打印梯度范数
        grad_norms = [np.linalg.norm(dW1), np.linalg.norm(db1), np.linalg.norm(dW2), np.linalg.norm(db2)]
        print(f"Epoch {epoch+1}, Grad Norms: {grad_norms}")

if rank == 0:
    print("\nFinal model parameters:")
    print("W1:", W1)
    print("b1:", b1)
    print("W2:", W2)
    print("b2:", b2)
    # 训练集RMSE
    def predict(X):
        a1 = np.tanh(X.dot(W1) + b1)
        return a1.dot(W2) + b2
    y_train_pred = predict(X_train)
    y_test_pred = predict(X_test)
    train_rmse = np.sqrt(np.mean((y_train_pred - y_train) ** 2))
    test_rmse = np.sqrt(np.mean((y_test_pred - y_test) ** 2))
    print(f"Train RMSE: {train_rmse:.4f}")
    print(f"Test RMSE: {test_rmse:.4f}")

MPI.Finalize()


# # 简单的神经网络回归示例，使用 NumPy 实现前向和反向传播，优化 L1 损失
# # 激活函数（tanh）
# def activation(x):
#     return np.tanh(x)

# # 激活函数的导数
# def activation_deriv(x):
#     return 1 - np.tanh(x) ** 2

# # 设置随机种子
# np.random.seed(42)

# # 模拟数据
# n_samples, input_dim, hidden_dim, output_dim = 1000, 20, 64, 1
# X = np.random.randn(n_samples, input_dim)
# true_W = np.random.randn(input_dim, output_dim)
# y = X.dot(true_W) + 0.1 * np.random.randn(n_samples, output_dim)  # 线性目标 + 噪声

# # 初始化参数
# W1 = np.random.randn(input_dim, hidden_dim) * 0.01
# b1 = np.zeros((1, hidden_dim))
# W2 = np.random.randn(hidden_dim, output_dim) * 0.01
# b2 = np.zeros((1, output_dim))

# # 超参数
# lr = 0.01
# epochs = 50
# batch_size = 64

# # 训练
# for epoch in range(epochs):
#     # 打乱数据
#     indices = np.random.permutation(n_samples)
#     X, y = X[indices], y[indices]

#     for i in range(0, n_samples, batch_size):
#         X_batch = X[i:i+batch_size]
#         y_batch = y[i:i+batch_size]

#         # 前向传播
#         z1 = X_batch.dot(W1) + b1
#         a1 = activation(z1)
#         y_pred = a1.dot(W2) + b2

#         # L1 损失及其导数
#         loss = np.mean(np.abs(y_pred - y_batch))
#         dL_dy = np.sign(y_pred - y_batch) / batch_size

#         # 反向传播
#         dW2 = a1.T.dot(dL_dy)
#         db2 = np.sum(dL_dy, axis=0, keepdims=True)

#         da1 = dL_dy.dot(W2.T)
#         dz1 = da1 * activation_deriv(z1)
#         dW1 = X_batch.T.dot(dz1)
#         db1 = np.sum(dz1, axis=0, keepdims=True)

#         # 参数更新
#         W1 -= lr * dW1
#         b1 -= lr * db1
#         W2 -= lr * dW2
#         b2 -= lr * db2

#     print(f"Epoch {epoch+1}/{epochs}, Loss={loss:.4f}")