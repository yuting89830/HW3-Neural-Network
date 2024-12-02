# https://pytorch.org/tutorials/beginner/basics/intro.html

from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # TODO: Define the layers of the model
        # 1. Fully Connected Layers Only !!!
        # 2. Try different number of layers and neurons
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
        # 3. (Bonus) Try convolutional layers
        # pass

    def forward(self, x):
        # TODO: Define the forward pass of the model
        # pass
        x = x.view(-1, 28 * 28)  # Flatten the image
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # Output layer
        return x
    
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)  # 第一層卷積
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)  # 第二層卷積
        self.fc1 = nn.Linear(64 * 7 * 7, 128)  # 全連接層
        self.fc2 = nn.Linear(128, 10)  # 輸出層

    def forward(self, x):
        x = F.relu(self.conv1(x))  # 卷積層1
        x = F.max_pool2d(x, 2)  # 最大池化層
        x = F.relu(self.conv2(x))  # 卷積層2
        x = F.max_pool2d(x, 2)  # 最大池化層
        x = x.view(-1, 64 * 7 * 7)  # 展平
        x = F.relu(self.fc1(x))  # 全連接層1
        x = self.fc2(x)  # 輸出層
        return x

def train(args, model, train_loader, optimizer, epoch):
    # Set the model to training mode
    model.train()
    pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Training]")
    total_loss = 0
    correct = 0
    # TODO: Define the training loop
    # for batch_idx, (data, target) in enumerate(train_loader):
    for data, target in pbar:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()

        pbar.set_postfix(loss=loss.item(), acc=100. * correct / len(train_loader.dataset))

    avg_loss = total_loss / len(train_loader)
    accuracy = 100. * correct / len(train_loader.dataset)
    return avg_loss, accuracy
        # pass


def test(model, test_loader):
    # Set the model to evaluation mode
    model.eval()
    test_loss = 0
    correct = 0
    pbar = tqdm(test_loader, desc="[Testing]")
    # TODO: Define the testing loop
    with torch.no_grad():
        # for data, target in test_loader:
        #     pass
        for data, target in pbar:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    # Log the testing status
    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f"\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n")
    return test_loss, accuracy

def validate(model, val_loader):
    # 模型進入驗證模式
    model.eval()
    val_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            val_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    val_loss /= len(val_loader.dataset)
    accuracy = 100. * correct / len(val_loader.dataset)
    return val_loss, accuracy


def plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies, test_accuracies, output_path, output_file):
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(12, 6))

    # 繪製損失
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label="Training Loss")
    plt.plot(epochs, val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss")
    plt.legend()

    # 繪製準確率
    plt.subplot(1, 2, 2)
    plt.plot(epochs, test_accuracies, label="Testing Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Testing Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"{output_path}/{output_file}.png")
    plt.close()



def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=0.06, metavar='LR',
                        help='learning rate')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--model-type', type=str, default='CNN', choices=['NN', 'CNN'],
                        help="Choose the model type")
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--output-file', type=str, default='output', help='File to save the text and plot output')

    args = parser.parse_args()
    # class Args:
    #     def __init__(self):
    #         self.batch_size = 64
    #         self.test_batch_size = 1000
    #         self.epochs = 20
    #         self.lr = 0.01
    #         self.gamma = 0.7
    #         self.dry_run = False
    #         self.seed = 1
    #         self.log_interval = 10
    #         self.save_model = False
    # args = Args()

    torch.manual_seed(args.seed)
    if not os.path.exists(args.output_file):
        os.makedirs(f"outputs/{args.output_file}")
        output_path = f"outputs/{args.output_file}"
    # TODO: Tune the batch size to see different results
    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}

    # Set transformation for the dataset
    # TODO: (Bonus) Change different dataset and transformations (data augmentation)
    # https://pytorch.org/vision/stable/datasets.html
    # https://pytorch.org/vision/main/transforms.html
    # e.g. CIFAR-10, Caltech101, etc. 
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    dataset1 = datasets.MNIST('../data', train=True, download=True,
                    transform=transform)
    dataset2 = datasets.MNIST('../data', train=False,
                    transform=transform)
    dataset_size = len(dataset1)  # dataset1 為完整的 MNIST 訓練資料集
    val_size = int(0.1 * dataset_size)  # 驗證集大小（10%）
    train_size = dataset_size - val_size  # 訓練集大小（90%）

    # 使用 random_split 進行分割
    train_dataset, val_dataset = torch.utils.data.random_split(dataset1, [train_size, val_size])

    train_loader = torch.utils.data.DataLoader(train_dataset,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)
    val_loader = torch.utils.data.DataLoader(val_dataset, **test_kwargs)

    if args.model_type == 'CNN':
        model = CNN().to(device)
    else:
        model = Net().to(device)

    # TODO: Tune the learning rate / optimizer to see different results
    optimizer = optim.SGD(model.parameters(), lr=args.lr)

    # TODO: Tune the learning rate scheduler to see different results
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    train_losses = []
    val_losses = []
    test_losses = []
    train_accuracies = []
    val_accuracies = []
    test_accuracies = []

    with open(f"{output_path}/{args.output_file}.txt", 'w') as f:
        for epoch in range(1, args.epochs + 1):
            train_loss, train_accuracy = train(args, model, train_loader, optimizer, epoch)
            val_loss, val_accuracy = validate(model, val_loader)
            test_loss, test_accuracy = test(model, test_loader)
            scheduler.step()

            # Log results to the output file
            f.write(f"Epoch {epoch}: "
                    f"Train Loss: {train_loss:.4f}| Train Accuracy: {train_accuracy:.2f}% , "
                    f"Val Loss: {val_loss:.4f}| Val Accuracy: {val_accuracy:.2f}% , "
                    f"Test Loss: {test_loss:.4f}| Test Accuracy: {test_accuracy:.2f}%\n")

            # Record losses and accuracies
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            test_losses.append(test_loss)
            train_accuracies.append(train_accuracy)
            val_accuracies.append(val_accuracy)
            test_accuracies.append(test_accuracy)

        # Plot and save the metrics
        plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies, test_accuracies, output_path, args.output_file)

    if args.save_model:
        torch.save(model.state_dict(), "mnist.pt")


if __name__ == '__main__':
    main()