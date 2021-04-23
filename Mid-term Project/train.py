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


# Load data
# def data_loader(root, data_size, batch_size, image_size, num_workers, train_ratio, valid_ratio):

#     transform = transforms.Compose([
#         transforms.Resize(image_size),
#         transforms.CenterCrop(image_size),
#         transforms.ToTensor(),
#         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
#     ])

#     dataset = datasets.ImageFolder(root=root, transform=transform)

#     total_size = len(dataset)
#     data_size = min(data_size, total_size)
#     train_size = int(data_size * train_ratio)
#     valid_size = int(data_size * valid_ratio)

#     index = list(range(total_size))
#     np.random.shuffle(index)

#     train_sampler = SubsetRandomSampler(index[:train_size])
#     valid_sampler = SubsetRandomSampler(index[train_size:train_size + valid_size])
#     test_sampler = SubsetRandomSampler(index[train_size + valid_size:data_size])

#     train_loader = DataLoader(dataset=dataset, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers)
#     valid_loader = DataLoader(dataset=dataset, batch_size=batch_size, sampler=valid_sampler, num_workers=num_workers)
#     test_loader = DataLoader(dataset=dataset, batch_size=batch_size, sampler=test_sampler, num_workers=num_workers)

#     return {'train': train_loader, 'valid': valid_loader, 'test': test_loader}


# Train
def train(dataloader, network, criterion, optimizer, num_epochs, log_show, log_root, device):

    train_loss_history, train_acc_history = [], []
    valid_loss_history, valid_acc_history = [], []
    best_network = copy.deepcopy(network.state_dict())
    best_acc = 0.0

    start = time.time()

    with open(f'{log_root}/train.log', 'w') as logging:

        for epoch in range(1, num_epochs + 1):

            # Training
            network.train()
            batch, num_batchces = 0, len(dataloader['train'])
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
                train_loss += loss.item()
                # Accuracy
                train_correct += torch.sum(torch.max(prediction, 1)[1] == label)
                train_total += len(label)
                # Log
                if batch % log_show == 0:
                    train_show_loss = train_loss / train_total
                    train_show_acc = float(train_correct) / train_total
                    print(f'Epoch [{epoch}/{num_epochs}] | Batch [{batch}/{num_batchces}] | Train Loss: {train_show_loss:.6f} | Train Acc: {train_show_acc:.4f}')

            train_epoch_loss = train_loss / train_total
            train_epoch_acc = float(train_correct) / train_total
            train_loss_history.append(train_epoch_loss)
            train_acc_history.append(train_epoch_acc)
            print(f'Epoch [{epoch}/{num_epochs}] | Train Loss: {train_epoch_loss:.6f} | Train Acc: {train_epoch_acc:.4f}')

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
                    valid_loss += loss.item()
                    # Accuracy
                    valid_correct += torch.sum(torch.max(prediction, 1)[1] == label)
                    valid_total += len(label)

            valid_epoch_loss = valid_loss / valid_total
            valid_epoch_acc = float(valid_correct) / valid_total
            valid_loss_history.append(valid_epoch_loss)
            valid_acc_history.append(valid_epoch_acc)
            print(f'Epoch [{epoch}/{num_epochs}] | Valid Loss: {valid_epoch_loss:.6f} | Valid Acc: {valid_epoch_acc:.4f}')

            if valid_epoch_acc > best_acc:
                best_network = copy.deepcopy(network.state_dict())
                best_acc = valid_epoch_acc

            logging.write(f'{epoch},{train_epoch_loss:.6f},{valid_epoch_loss:.6f},{train_epoch_acc:.4f},{valid_epoch_acc:.4f}\n')
            print('-' * 80)

    end = time.time()
    elapse = end - start
    print(f'Training Elapsed Time: {elapse // 3600:.0f} h {elapse % 3600 // 60:.0f} m {elapse % 60:.0f} s')
    print('-' * 80)
    print(f'Best Valid Acc: {best_acc:.4f}')

    network.load_state_dict(best_network)
    history = {
        'train_loss_curve': train_loss_history,
        'train_acc_curve': train_acc_history,
        'valid_loss_curve': valid_loss_history,
        'valid_acc_curve': valid_acc_history,
        'best_acc': best_acc
    }
    return network, history


# Test
def test(dataloader, network, device):

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
    print(f'Test Acc: {test_acc:.4f}')
    print('-' * 80)


# Save model
def save_model(root, model, depth, data_size):
    torch.save(model.state_dict(), f'{root}/ResNet{depth}_{data_size}.pth')
    print(f'Model ResNet{depth}_{data_size}.pth saved!')


# Visual
def visual(root, history):
    pass


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
    # dataloader = data_loader(
    #     root=config.data_root, data_size=config.data_size, batch_size=config.batch_size, image_size=config.image_size,
    #     num_workers=config.num_workers, train_ratio=config.train_ratio, valid_ratio=config.valid_ratio
    # )
    # TODO: dataloader
    dataloader = {'train': None, 'valid': None, 'test': None}
    # Network
    network = resnet(config.depth).to(device=device)
    # Criterion
    criterion = nn.CrossEntropyLoss()
    # Optimizer
    optimizer = optim.Adam(params=network.parameters(), lr=config.learning_rate, weight_decay=config.penalty)

    # Train
    model, history = train(
        dataloader, network, criterion, optimizer, num_epochs=config.epoch,
        log_show=config.log_show, log_root=config.log_root, device=device
    )
    # Test
    test(dataloader, model, device=device)

    # Save
    save_model(root=config.model_root, model=model, depth=config.depth, data_size=config.data_size)
    # Visual
    visual(root=config.visual_root, history=history)


if __name__ == '__main__':

    # Parser
    parser = argparse.ArgumentParser()
    # Device
    parser.add_argument('--device', type=str, default=('cuda' if torch.cuda.is_available() else 'cpu'))
    parser.add_argument('--cuda', type=int, default=(0 if torch.cuda.is_available() else -1))
    # Random seed
    parser.add_argument('--seed', type=int, default=2021)
    # Path
    parser.add_argument('--data_root', type=str, default='../dataset')
    parser.add_argument('--log_root', type=str, default='../log')
    parser.add_argument('--model_root', type=str, default='../model')
    parser.add_argument('--visual_root', type=str, default='../visual')
    # Data parameter
    parser.add_argument('--data_size', type=int, default=70000)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--image_size', type=int, default=28)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--train_ratio', type=float, default=0.6)
    parser.add_argument('--valid_ratio', type=float, default=0.2)
    # Network parameter
    parser.add_argument('--depth', type=int, default=18)
    # Optimizer parameter
    parser.add_argument('--learning_rate', type=float, default=0.1)
    parser.add_argument('--penalty', type=float, default=0.0001)
    # Training parameter
    parser.add_argument('--epoch', type=int, default=200)
    parser.add_argument('--log_show', type=int, default=10)
    # Configuration
    CONFIG = parser.parse_args()

    # Main
    main(CONFIG)
