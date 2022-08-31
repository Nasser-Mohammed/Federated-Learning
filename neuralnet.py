
from torch import nn

def create_model():
    model = nn.Sequential(
        nn.Conv2d(in_channels = 1, out_channels = 32, kernel_size = 3, stride = 1),
        nn.ReLU(),
        nn.MaxPool2d(2, stride = 2),

        nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 5, stride = 1), 
        nn.ReLU(),
        nn.MaxPool2d(2, stride = 2),

        nn.Flatten(),
        nn.Linear(1024, 200),
        nn.ReLU(),
        nn.Linear(200, 84),
        nn.ReLU(),
        nn.Linear(84, 10)
)
    return model




