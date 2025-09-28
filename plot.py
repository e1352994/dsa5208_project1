import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('/Users/vald/Downloads/DSA5208分布式计算/assignment1/sgd_training_losses.csv')
data.head()
data['nprocess&batch'] = data['num_of_process'].astype(str) + '-' + data['batch_size'].astype(str)
data.head()

for activation in data['activation'].unique():
    subset = data[data['activation'] == 'tanh']
    plt.figure(figsize=(8, 6))
    for key, group in subset.groupby('nprocess&batch'):
        plt.plot(group['epoch'], group['loss'], label=key, marker='o', alpha=0.5)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Loss vs Epoch (activation={activation})')
    plt.legend(title='number of process & batch size')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'loss_vs_epoch_{activation}.png')
    plt.show()

datafin = pd.read_excel('/Users/vald/Downloads/DSA5208分布式计算/assignment1/minloss.xlsx')
datafin.head()

plt.figure(figsize=(8, 6))
plt.plot(datafin['epoch'], datafin['loss'], marker='o', color='red')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.title('minimum loss vs epoch')
plt.grid(True)
plt.tight_layout()
plt.savefig('min_loss_vs_epoch.png')
plt.show()
