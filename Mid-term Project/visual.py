import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=100)
parser.add_argument('--breaks', type=int, default=10)
config = parser.parse_args()

num_epochs = config.epoch
breaks = config.breaks

with open(f'../log/train_{num_epochs}.log', 'r') as f:
    logging = f.readlines()
    history = {'train_loss_curve': [], 'valid_loss_curve': [], 'train_acc_curve': [], 'valid_acc_curve': []}
    for line in logging:
        h = line[:-1].split(',')
        history['train_loss_curve'].append(float(h[1]))
        history['valid_loss_curve'].append(float(h[2]))
        history['train_acc_curve'].append(float(h[3]))
        history['valid_acc_curve'].append(float(h[4]))

plt.figure()
plt.grid(ls='--')
plt.plot(list(range(1, num_epochs + 1)), history['train_loss_curve'], 'b-', linewidth=1, label='Train Loss')
plt.plot(list(range(1, num_epochs + 1)), history['valid_loss_curve'], 'r-', linewidth=1, label='Valid Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.xticks(list(range(0, num_epochs + 1, breaks)))
plt.legend()
plt.savefig(f'loss_curve_{num_epochs}.png')
plt.close('all')
print(f'loss_curve_{num_epochs}.png saved!')

plt.figure()
plt.grid(ls='--')
plt.plot(list(range(1, num_epochs + 1)), history['train_acc_curve'], 'b-', linewidth=1, label='Train Acc.')
plt.plot(list(range(1, num_epochs + 1)), history['valid_acc_curve'], 'r-', linewidth=1, label='Valid Acc.')
plt.xlabel('Epoch')
plt.ylabel('Acc')
plt.xticks(list(range(0, num_epochs + 1, breaks)))
plt.legend()
plt.savefig(f'acc_curve_{num_epochs}.png')
plt.close('all')
print(f'acc_curve_{num_epochs}.png saved!')