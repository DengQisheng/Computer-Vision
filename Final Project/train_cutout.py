import argparse
import copy
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, SubsetRandomSampler

from ResNet import resnet


# Set Random Seed
def set_random_seeds(seed_value=0, device_type='cpu'):
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if device_type != 'cpu':
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# Cutout transform
class Cutout(object):

    def __init__(self, size):
        self.size = size

    def __call__(self, img):
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.size // 2, 0, h)
        y2 = np.clip(y + self.size // 2, 0, h)
        x1 = np.clip(x - self.size // 2, 0, w)
        x2 = np.clip(x + self.size // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img


# Load data
def data_loader(root, batch_size, hole_size, num_workers):

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
        Cutout(size=hole_size)
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))
    ])

    train_set = datasets.CIFAR10(root=root, train=True, transform=transform_train)
    test_set = datasets.CIFAR10(root=root, train=False, transform=transform_test)

    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    valid_loader = DataLoader(
        dataset=test_set, batch_size=batch_size, sampler=SubsetRandomSampler(list(range(0, 5000))), num_workers=num_workers
    )
    test_loader = DataLoader(
        dataset=test_set, batch_size=batch_size, sampler=SubsetRandomSampler(list(range(5000, 10000))), num_workers=num_workers
    )

    return train_loader, valid_loader, test_loader


# Train
def train(train_loader, valid_loader, network, criterion, optimizer, scheduler, num_epochs, log_show, log_root, device, depth):

    train_loss_history, train_acc_history = [], []
    valid_loss_history, valid_acc_history = [], []
    best_network_wts = copy.deepcopy(network.state_dict())
    best_acc = 0.0

    start = time.time()

    with open(f'{log_root}/ResNet{depth}_train_cutout.log', 'w') as logging:

        for epoch in range(1, num_epochs + 1):

            # Training
            network.train()
            batch, num_batches = 0, len(train_loader)
            train_loss, train_correct, train_total = 0.0, 0, 0
            for patch, label in train_loader:
                # Data
                patch = patch.to(device)
                label = label.to(device)
                batch += 1
                # Forward
                prediction = network(patch)
                loss = criterion(prediction, label)
                # Backprop
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # Loss
                train_loss += loss.item() * patch.size(0)
                # Accuracy
                train_correct += torch.sum(torch.max(prediction, 1)[1] == label)
                train_total += len(label)
                # Log
                if batch % log_show == 0:
                    train_show_loss = train_loss / train_total
                    train_show_acc = float(train_correct) / train_total
                    print(f'Epoch [{epoch}/{num_epochs}] | Batch [{batch}/{num_batches}] | Train Loss: {train_show_loss:.6f} | Train Acc: {train_show_acc:.4f}')

            train_epoch_loss = train_loss / train_total
            train_epoch_acc = float(train_correct) / train_total
            train_loss_history.append(train_epoch_loss)
            train_acc_history.append(train_epoch_acc)
            print(f'Epoch [{epoch}/{num_epochs}] | Train Loss: {train_epoch_loss:.6f} | Train Acc: {train_epoch_acc:.4f}')

            # Validation
            network.eval()
            valid_loss, valid_correct, valid_total = 0.0, 0, 0
            with torch.no_grad():
                for patch, label in valid_loader:
                    # Data
                    patch = patch.to(device)
                    label = label.to(device)
                    # Forward
                    prediction = network(patch)
                    loss = criterion(prediction, label)
                    # Loss
                    valid_loss += loss.item() * patch.size(0)
                    # Accuracy
                    valid_correct += torch.sum(torch.max(prediction, 1)[1] == label)
                    valid_total += len(label)

            valid_epoch_loss = valid_loss / valid_total
            valid_epoch_acc = float(valid_correct) / valid_total
            valid_loss_history.append(valid_epoch_loss)
            valid_acc_history.append(valid_epoch_acc)
            print(f'Epoch [{epoch}/{num_epochs}] | Valid Loss: {valid_epoch_loss:.6f} | Valid Acc: {valid_epoch_acc:.4f}')

            if valid_epoch_acc > best_acc:
                best_network_wts = copy.deepcopy(network.state_dict())
                best_acc = valid_epoch_acc

            logging.write(f'{epoch},{train_epoch_loss:.6f},{valid_epoch_loss:.6f},{train_epoch_acc:.4f},{valid_epoch_acc:.4f}\n')
            print('-' * 80)

            scheduler.step()

    end = time.time()
    elapse = end - start
    print(f'Training Elapsed Time: {elapse // 3600:.0f} h {elapse % 3600 // 60:.0f} m {elapse % 60:.0f} s')
    print('-' * 80)
    print(f'Best Valid Acc: {best_acc:.4f}')

    network.load_state_dict(best_network_wts)
    history = {
        'train_loss_curve': train_loss_history,
        'train_acc_curve': train_acc_history,
        'valid_loss_curve': valid_loss_history,
        'valid_acc_curve': valid_acc_history,
        'best_acc': best_acc
    }
    return network, history


# Test
def test(test_loader, network, device):

    network.eval()
    test_correct, test_total = 0, 0
    with torch.no_grad():
        for patch, label in test_loader:
            # Data
            patch = patch.to(device)
            label = label.to(device)
            # Predict
            prediction = network(patch)
            # Accuracy
            test_correct += torch.sum(torch.max(prediction, 1)[1] == label)
            test_total += len(label)

    test_acc = float(test_correct) / test_total
    print(f'Test Acc: {test_acc:.4f}')
    print('-' * 80)


# Save model
def save_model(root, model, depth):
    torch.save(model.state_dict(), f'{root}/ResNet{depth}_cutout.pth')
    print(f'Model ResNet{depth}_cutout.pth saved!')


# Visual
def visual(root, history, num_epochs, depth):

    # Loss
    plt.figure()
    plt.grid(ls='--')
    plt.plot(range(1, num_epochs + 1), history['train_loss_curve'], 'b-', linewidth=1, label='Train Loss')
    plt.plot(range(1, num_epochs + 1), history['valid_loss_curve'], 'r-', linewidth=1, label='Valid Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.xticks(list(range(0, num_epochs + 1, 25)))
    plt.legend()
    plt.savefig(f'{root}/ResNet{depth}_loss_curve_cutout.png')
    plt.close('all')
    print(f'ResNet{depth}_loss_curve_cutout.png saved!')

    # Accuracy
    plt.figure()
    plt.grid(ls='--')
    plt.plot(range(1, num_epochs + 1), history['train_acc_curve'], 'b-', linewidth=1, label='Train Acc.')
    plt.plot(range(1, num_epochs + 1), history['valid_acc_curve'], 'r-', linewidth=1, label='Valid Acc.')
    plt.xlabel('Epoch')
    plt.ylabel('Acc')
    plt.xticks(list(range(0, num_epochs + 1, 25)))
    plt.legend()
    plt.savefig(f'{root}/ResNet{depth}_acc_curve_cutout.png')
    plt.close('all')
    print(f'ResNet{depth}_acc_curve_cutout.png saved!')


# Main
def main(config):

    # Log
    print()
    print('\n'.join([f'{k}: {v}' for k, v in vars(config).items()]))
    print('-' * 80)

    # Device
    device = torch.device(f'{config.device}:{config.cuda}' if torch.cuda.is_available() else 'cpu')

    # Random seed
    set_random_seeds(seed_value=config.seed, device_type=device.type)

    # Dataset
    train_loader, valid_loader, test_loader = data_loader(
        root=config.data_root, batch_size=config.batch_size, hole_size=config.hole_size, num_workers=config.num_workers
    )
    # Network
    network = resnet(config.depth).to(device=device)
    # Criterion
    criterion = nn.CrossEntropyLoss()
    # Optimizer
    optimizer = optim.SGD(params=network.parameters(), lr=config.learning_rate, momentum=config.momentum, weight_decay=config.penalty)
    # Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epoch)

    # Train
    model, history = train(
        train_loader, valid_loader, network, criterion, optimizer, scheduler, num_epochs=config.epoch,
        log_show=config.log_show, log_root=config.log_root, device=device, depth=config.depth
    )
    # Test
    test(test_loader, model, device=device)

    # Save
    save_model(root=config.model_root, model=model, depth=config.depth)
    # Visual
    visual(root=config.visual_root, history=history, num_epochs=config.epoch, depth=config.depth)


# Parser
if __name__ == '__main__':

    # Parser
    parser = argparse.ArgumentParser()
    # Device
    parser.add_argument('--device', type=str, default=('cuda' if torch.cuda.is_available() else 'cpu'))
    parser.add_argument('--cuda', type=int, default=(0 if torch.cuda.is_available() else -1))
    # Random seed
    parser.add_argument('--seed', type=int, default=2021)
    # Path
    parser.add_argument('--data_root', type=str, default='../data')
    parser.add_argument('--log_root', type=str, default='../log')
    parser.add_argument('--model_root', type=str, default='../model')
    parser.add_argument('--visual_root', type=str, default='../visual')
    # Data parameter
    parser.add_argument('--hole_size', type=int, default=16)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=8)
    # Network parameter
    parser.add_argument('--depth', type=int, default=18)
    # Optimizer parameter
    parser.add_argument('--learning_rate', type=float, default=1e-1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--penalty', type=float, default=5e-4)
    # Training parameter
    parser.add_argument('--epoch', type=int, default=250)
    parser.add_argument('--log_show', type=int, default=50)
    # Configuration
    config = parser.parse_args()

    # Main
    main(config)
