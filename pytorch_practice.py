import torch
import torch.nn as nn
import torch.nn.functional as F

linear = nn.Linear(128, 256)
input_data = torch.zeros((128))
print("--linear-------------------------")
print(linear)
print("--input_data-------------------------")
print(input_data)

print("--x = input_data-------------------------")
x = input_data
print(x)

print("--x = linear(x)-------------------------")
x = linear(x)
print(x)


print("--x = F.relu(x)-------------------------")
x = F.relu(x)
print(x)

print("--x.shape-------------------------")
print(x.shape)


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(128, 256)
        self.linear2 = nn.Linear(256, 10)
        
    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        return x


net = MLP()
print("--net-------------------------")
print(net)

y = net(input_data)
print("--y-------------------------")
print(y)

print("--y.shape-------------------------")
print(y.shape)


net = nn.Sequential(nn.Linear(128, 256), nn.ReLU(), nn.Linear(256, 10))
y = net(input_data)
print("--y-------------------------")
print(y)
print("--y.shape-------------------------")
print(y.shape)




input_data = torch.rand([32, 128])
y = net(input_data)
print("--y-------------------------")
print(y)
print("--y.shape-------------------------")
print(y.shape)