import warnings
from collections import OrderedDict
import flwr as fl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, Normalize, ToTensor
from tqdm import tqdm
import numpy as np
import pandas
# #############################################################################
# 1. Regular PyTorch pipeline: nn.Module, train, test, and DataLoader
# #############################################################################
warnings.filterwarnings("ignore", category=UserWarning)
# DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DEVICE = torch.device("cpu")

class CsvDataset(Dataset):
    def __init__(self,opt):
        if opt == 'train':
            self.filepath = "/home/lumiawjf/BFLC/BCFL/train.csv"
        elif opt == 'test':
            self.filepath = "/home/lumiawjf/BFLC/BCFL/test.csv"
        print(f"reading {opt} files!")

        #create columns name list and dict
        name = []
        for i in range(520):
            name.append("wap%d" %i)
        name.append("spaceID")
        dtype = dict.fromkeys(name, np.float32)
        #read csv
        df = pandas.read_csv(self.filepath, header = 0, index_col=0,
            encoding = 'utf-8',names = tuple(name), dtype = dtype,
            skip_blank_lines = True,)
        wap = df.iloc[:,0:520].values
        label = df.iloc[:,520].values
        self.x = torch.from_numpy(wap)
        self.y = torch.from_numpy(label)

    def __len__(self):
        return len(self.y)
    def __getitem__(self,index):
        return self.x[index], self.y[index]
    
class Net(nn.Module):
    def __init__(self) -> None:
        super(Net, self).__init__()
        self.fc1 = nn.Linear(520,10,bias=True)
        self.fc2 = nn.Linear(10,13,bias=True)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x))


def train(net, trainloader, epochs):
    """Train the model on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    for _ in range(epochs):
        for images, labels in tqdm(trainloader):
            optimizer.zero_grad()
            criterion(net(images.to(DEVICE)), labels.to(DEVICE).long()).backward()
            optimizer.step()

def test(net, testloader):
    """Validate the model on the test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    with torch.no_grad():
        for images, labels in tqdm(testloader):
            outputs = net(images.to(DEVICE))
            labels = labels.to(DEVICE)
            loss += criterion(outputs, labels.long()).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).type(torch.float).sum().item()
    accuracy = correct / len(testloader.dataset)
    return loss, accuracy


def load_data():
    trainset = CsvDataset('train')
    testset = CsvDataset('test')
    return DataLoader(trainset,batch_size=800, shuffle=True), DataLoader(testset)


# #############################################################################
# 2. Federation of the pipeline with Flower
# #############################################################################
# Load model and data (simple CNN, CIFAR-10)
net = Net().to(DEVICE)
trainloader, testloader = load_data()

# Define Flower client
class FlowerClient(fl.client.NumPyClient):
    def get_parameters(self):
        return [val.cpu().numpy() for _, val in net.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        train(net, trainloader, epochs=1)
        torch.save(net, "mnist_cnn.pt")
        return self.get_parameters(), len(trainloader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = test(net, testloader)
        return loss, len(testloader.dataset), {"accuracy": accuracy}

# Start the flower client
fl.client.start_numpy_client("[::]:8081", client=FlowerClient())
