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
from torch.utils.data import DataLoader, random_split

from ResNet import resnet as resnet_with_BN
from ResNet_without_BN import resnet as resnet_without_BN


# Set Random Seed
def set_random_seeds(seed_value=0, device_type='cpu'):
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if device_type != 'cpu':
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# Load data
def data_loader(root, batch_size, num_workers, valid_size):

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_set = datasets.mnist.MNIST(root, transform=transform, train=True)
    valid_set, test_set = random_split(datasets.mnist.MNIST(root, transform=transform, train=False), lengths=[valid_size, 10000 - valid_size])

    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    valid_loader = DataLoader(dataset=valid_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return {'train': train_loader, 'valid': valid_loader, 'test': test_loader}


# Train
def train(dataloader, network, criterion, optimizer, scheduler, num_epochs, log_show, log_root, device, BN, lr):

    train_loss_history, train_acc_history = [], []
    valid_loss_history, valid_acc_history = [], []
    loss_range = []
    best_network = copy.deepcopy(network.state_dict())
    best_acc = 0.0

    start = time.time()

    with open(f'{log_root}/train_BN_{BN}_lr_{lr}_{num_epochs}.log', 'w') as logging:

        for epoch in range(1, num_epochs + 1):

            # Training
            network.train()
            batch, num_batches = 0, len(dataloader['train'])
            train_loss, train_correct, train_total = 0.0, 0, 0
            for patch, label in dataloader['train']:
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
                loss_range.append(loss.item())
                train_loss += loss.item() * patch.size(0)
                # Accuracy
                train_correct += torch.sum(torch.max(prediction, 1)[1] == label)
                train_total += len(label)
                # Log
                if batch % log_show == 0:
                    train_show_loss = train_loss / train_total
                    train_show_acc = float(train_correct) / train_total
                    print(f'BN {BN} | lr {lr} | Epoch [{epoch}/{num_epochs}] | Batch [{batch}/{num_batches}] | Train Loss: {train_show_loss:.4f} | Train Acc: {train_show_acc:.4f}')

            train_epoch_loss = train_loss / train_total
            train_epoch_acc = float(train_correct) / train_total
            train_loss_history.append(train_epoch_loss)
            train_acc_history.append(train_epoch_acc)
            print(f'BN {BN} | lr {lr} | Epoch [{epoch}/{num_epochs}] | Train Loss: {train_epoch_loss:.4f} | Train Acc: {train_epoch_acc:.4f}')

            scheduler.step()

            # Validation
            network.eval()
            valid_loss, valid_correct, valid_total = 0.0, 0, 0
            with torch.no_grad():
                for patch, label in dataloader['valid']:
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
            print(f'BN {BN} | lr {lr} | Epoch [{epoch}/{num_epochs}] | Valid Loss: {valid_epoch_loss:.4f} | Valid Acc: {valid_epoch_acc:.4f}')

            if valid_epoch_acc > best_acc:
                best_network = copy.deepcopy(network.state_dict())
                best_acc = valid_epoch_acc

            logging.write(f'{epoch},{train_epoch_loss:.6f},{valid_epoch_loss:.6f},{train_epoch_acc:.4f},{valid_epoch_acc:.4f}\n')
            print('-' * 96)

    end = time.time()
    elapse = end - start
    print(f'Training Elapsed Time: {elapse // 3600:.0f} h {elapse % 3600 // 60:.0f} m {elapse % 60:.0f} s')
    print('-' * 96)
    print(f'Best Valid Acc: {best_acc:.4f}')
    print('-' * 96)

    network.load_state_dict(best_network)
    history = {
        'train_loss_curve': train_loss_history,
        'train_acc_curve': train_acc_history,
        'valid_loss_curve': valid_loss_history,
        'valid_acc_curve': valid_acc_history,
        'loss_range': loss_range
    }
    return network, history


# Test
def test(dataloader, network, device, BN, lr):

    network.eval()
    test_correct, test_total = 0, 0
    with torch.no_grad():
        for patch, label in dataloader['test']:
            # Data
            patch = patch.to(device)
            label = label.to(device)
            # Predict
            prediction = network(patch)
            # Accuracy
            test_correct += torch.sum(torch.max(prediction, 1)[1] == label)
            test_total += len(label)

    test_acc = float(test_correct) / test_total
    print(f'BN {BN} | lr {lr} | Test Acc: {test_acc:.4f}')
    print('-' * 96)


# Visual
def visual(root, matrix_with_BN, matrix_without_BN, num_epochs, learning_rate, breaks, ticks):

    for i, lr in enumerate(learning_rate):

        # Loss
        plt.figure()
        plt.grid(ls='--')
        plt.plot(range(1, num_epochs + 1), matrix_with_BN[i]['train_loss_curve'], 'b-', linewidth=1, label='Train Loss')
        plt.plot(range(1, num_epochs + 1), matrix_with_BN[i]['valid_loss_curve'], 'r-', linewidth=1, label='Valid Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.xticks(list(range(0, num_epochs + 1, 10)))
        plt.legend()
        plt.savefig(f'{root}/loss_curve_with_BN_{lr}_{num_epochs}.png')
        plt.close('all')
        print(f'loss_curve_with_BN_{lr}_{num_epochs}.png saved!')

        # Accuracy
        plt.figure()
        plt.grid(ls='--')
        plt.plot(range(1, num_epochs + 1), matrix_with_BN[i]['train_acc_curve'], 'b-', linewidth=1, label='Train Acc.')
        plt.plot(range(1, num_epochs + 1), matrix_with_BN[i]['valid_acc_curve'], 'r-', linewidth=1, label='Valid Acc.')
        plt.xlabel('Epoch')
        plt.ylabel('Acc')
        plt.xticks(list(range(0, num_epochs + 1, 10)))
        plt.legend()
        plt.savefig(f'{root}/acc_curve_with_BN_{lr}_{num_epochs}.png')
        plt.close('all')
        print(f'acc_curve_with_BN_{lr}_{num_epochs}.png saved!')

        # Loss
        plt.figure()
        plt.grid(ls='--')
        plt.plot(range(1, num_epochs + 1), matrix_without_BN[i]['train_loss_curve'], 'b-', linewidth=1, label='Train Loss')
        plt.plot(range(1, num_epochs + 1), matrix_without_BN[i]['valid_loss_curve'], 'r-', linewidth=1, label='Valid Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.xticks(list(range(0, num_epochs + 1, 10)))
        plt.legend()
        plt.savefig(f'{root}/loss_curve_without_BN_{lr}_{num_epochs}.png')
        plt.close('all')
        print(f'loss_curve_without_BN_{lr}_{num_epochs}.png saved!')

        # Accuracy
        plt.figure()
        plt.grid(ls='--')
        plt.plot(range(1, num_epochs + 1), matrix_without_BN[i]['train_acc_curve'], 'b-', linewidth=1, label='Train Acc.')
        plt.plot(range(1, num_epochs + 1), matrix_without_BN[i]['valid_acc_curve'], 'r-', linewidth=1, label='Valid Acc.')
        plt.xlabel('Epoch')
        plt.ylabel('Acc')
        plt.xticks(list(range(0, num_epochs + 1, 10)))
        plt.legend()
        plt.savefig(f'{root}/acc_curve_without_BN_{lr}_{num_epochs}.png')
        plt.close('all')
        print(f'acc_curve_without_BN_{lr}_{num_epochs}.png saved!')

    # Landscape
    min_with_BN = np.log(1 + np.min([matrix_with_BN[i]['loss_range'] for i in range(len(learning_rate))], axis=0))
    max_with_BN = np.log(1 + np.max([matrix_with_BN[i]['loss_range'] for i in range(len(learning_rate))], axis=0))
    min_without_BN = np.log(1 + np.min([matrix_without_BN[i]['loss_range'] for i in range(len(learning_rate))], axis=0))
    max_without_BN = np.log(1 + np.max([matrix_without_BN[i]['loss_range'] for i in range(len(learning_rate))], axis=0))
    steps = len(min_with_BN)
    plt.figure(figsize=(15, 10), dpi=240)
    plt.grid(ls='--')
    plt.plot(range(steps)[::breaks], min_with_BN[::breaks], color='#4169E1', linestyle='-', linewidth=1, alpha=0.8)
    plt.plot(range(steps)[::breaks], max_with_BN[::breaks], color='#4169E1', linestyle='-', linewidth=1, alpha=0.8)
    plt.plot(range(steps)[::breaks], min_without_BN[::breaks], color='#DB7093', linestyle='-', linewidth=1, alpha=0.8)
    plt.plot(range(steps)[::breaks], max_without_BN[::breaks], color='#DB7093', linestyle='-', linewidth=1, alpha=0.8)
    plt.fill_between(range(steps)[::breaks], min_with_BN[::breaks], max_with_BN[::breaks], facecolor='#6495ED', label='ResNet with BN', alpha=0.8)
    plt.fill_between(range(steps)[::breaks], min_without_BN[::breaks], max_without_BN[::breaks], facecolor='#FFB6C1', label='ResNet without BN', alpha=0.8)
    plt.xlabel('Step')
    plt.ylabel('Log(1 + Loss)')
    plt.xticks(list(range(0, steps + 1, ticks)))
    plt.legend()
    plt.savefig(f'{root}/loss_landscape_{num_epochs}.png')
    plt.close('all')
    print(f'loss_landscape_{num_epochs}.png saved!')


# Main
def main(config):

    # Log
    print()
    print('\n'.join([f'{k}: {v}' for k, v in vars(config).items()]))
    print('-' * 96)

    # Device
    device = torch.device(f'{config.device}:{config.cuda}' if torch.cuda.is_available() else 'cpu')

    # Random seed
    set_random_seeds(seed_value=config.seed, device_type=device.type)

    # Dataset
    dataloader = data_loader(
        root=config.data_root, batch_size=config.batch_size,
        num_workers=config.num_workers, valid_size=config.valid_size
    )

    matrix_with_BN, matrix_without_BN = [], []
    for lr in config.learning_rate:

        # Network
        network_with_BN = resnet_with_BN(config.depth).to(device=device)
        network_without_BN = resnet_without_BN(config.depth).to(device=device)
        # Criterion
        criterion_with_BN = nn.CrossEntropyLoss()
        criterion_without_BN = nn.CrossEntropyLoss()
        # Optimizer
        optimizer_with_BN = optim.Adam(params=network_with_BN.parameters(), lr=lr, weight_decay=config.penalty)
        optimizer_without_BN = optim.Adam(params=network_without_BN.parameters(), lr=lr, weight_decay=config.penalty)
        # Scheduler
        scheduler_with_BN = optim.lr_scheduler.MultiStepLR(optimizer_with_BN, milestones=config.milestones, gamma=config.gamma)
        scheduler_without_BN = optim.lr_scheduler.MultiStepLR(optimizer_without_BN, milestones=config.milestones, gamma=config.gamma)

        # Train
        model_with_BN, history_with_BN = train(
            dataloader, network_with_BN, criterion_with_BN, optimizer_with_BN, scheduler_with_BN, num_epochs=config.epoch,
            log_show=config.log_show, log_root=config.log_root, device=device, BN=True, lr=lr
        )
        model_without_BN, history_without_BN = train(
            dataloader, network_without_BN, criterion_without_BN, optimizer_without_BN, scheduler_without_BN, num_epochs=config.epoch,
            log_show=config.log_show, log_root=config.log_root, device=device, BN=False, lr=lr
        )
        matrix_with_BN.append(history_with_BN)
        matrix_without_BN.append(history_without_BN)
        # Test
        test(dataloader, model_with_BN, device=device, BN=True, lr=lr)
        test(dataloader, model_without_BN, device=device, BN=False, lr=lr)

    # Visual
    visual(
        root=config.visual_root, matrix_with_BN=matrix_with_BN, matrix_without_BN=matrix_without_BN,
        num_epochs=config.epoch, learning_rate=config.learning_rate, breaks=config.breaks, ticks=config.ticks
    )


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
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--valid_size', type=int, default=5000)
    # Network parameter
    parser.add_argument('--depth', type=int, default=18)
    # Optimizer parameter
    parser.add_argument('--learning_rate', type=list, default=[0.005, 0.001, 0.0005, 0.0001])
    parser.add_argument('--penalty', type=float, default=0.00001)
    parser.add_argument('--milestones', type=list, default=[10, 30, 50, 70, 90])
    parser.add_argument('--gamma', type=float, default=0.8)
    # Training parameter
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--log_show', type=int, default=100)
    # Visualization parameter
    parser.add_argument('--breaks', type=int, default=32)
    parser.add_argument('--ticks', type=int, default=5000)
    # Configuration
    CONFIG = parser.parse_args()

    # Main
    main(CONFIG)
