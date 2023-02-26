import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import pandas
import numpy as np
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
warnings.filterwarnings("ignore", category=UserWarning)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class CsvDataset(Dataset):
    def __init__(self,opt):
        if opt == 'train':
            self.filepath = "train.csv"
        elif opt == 'test':
            self.filepath = "test.csv"
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
    
def train(net, trainloader):
    net.train()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(),lr=0.01)
    size = len(trainloader.dataset)
    for batch , (data,labels) in enumerate(trainloader):
        data   = data.to(DEVICE)
        labels = labels.to(DEVICE)
        pred = net(data)
        loss = criterion(pred,labels.long())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(data)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(net, testloader):
    net.eval()
    size = len(testloader.dataset)
    num_batches = len(testloader)
    criterion = torch.nn.CrossEntropyLoss()
    correct, test_loss = 0, 0.0
    with torch.no_grad():
        for data, labels in tqdm(testloader):
            data   = data.to(DEVICE)
            labels = labels.to(DEVICE)
            pred = net(data)
            test_loss += criterion(pred,labels.long()).item()
            correct += (pred.argmax(1) == labels).type(torch.float).sum().item()
    test_loss /= num_batches
    correct   /= size
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

def load_data():
    trainset = CsvDataset('train')
    testset = CsvDataset('test')
    return DataLoader(trainset, batch_size=100, shuffle=True), DataLoader(testset)

net = Net().to(DEVICE)
trainloader, testloader = load_data()
epochs = 20
for t in range(epochs):
    print(f"Epoch {t+1} \n ------------------------------")
    train(net,trainloader)
    test(net,testloader)
print("Done!")