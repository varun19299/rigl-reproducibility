import torch
from torch import nn
from torch.nn import functional as F


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.cl1 = nn.Linear(25, 60)
        self.cl2 = nn.Linear(60, 16)
        self.fc1 = nn.Linear(16, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.relu(self.cl1(x))
        x = F.relu(self.cl2(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.log_softmax(self.fc3(x), dim=1)
        return x


activation = {}


def get_activation(name):
    def hook(model, input, output):
        activation[name] = input[0].detach()

    return hook


model = MyModel()
model.fc3.register_forward_hook(get_activation("fc3"))
x = torch.randn(1, 25)
output = model(x)
print(activation["fc3"].shape)
