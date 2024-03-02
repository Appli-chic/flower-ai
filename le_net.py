from torch import flatten
from torch.nn import Conv2d
from torch.nn import Linear
from torch.nn import LogSoftmax
from torch.nn import MaxPool2d
from torch.nn import Module
from torch.nn import ReLU


class LeNet(Module):
    def __init__(self, classes):
        super(LeNet, self).__init__()

        # Initialize the first layers of Convolutional, activation function and pooling
        self.conv1 = Conv2d(in_channels=3, out_channels=20, kernel_size=(5, 5))
        self.relu1 = ReLU()
        self.maxpool1 = MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        # Initialize the seconds layers of Convolutional, activation function and pooling
        self.conv2 = Conv2d(in_channels=20, out_channels=50, kernel_size=(5, 5))
        self.relu2 = ReLU()
        self.maxpool2 = MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        # Initialize the fully connected layers
        input_size = 250
        conv1_output_size = (input_size - (5 - 1)) // 2  # size after first conv and pool
        conv2_output_size = (conv1_output_size - (5 - 1)) // 2  # size after second conv and pool
        fc1_in_features = 50 * conv2_output_size * conv2_output_size
        self.fc1 = Linear(in_features=fc1_in_features, out_features=500)
        self.relu3 = ReLU()

        # initialize softmax classifier
        self.fc2 = Linear(in_features=500, out_features=classes)
        self.logSoftmax = LogSoftmax(dim=1)

    def forward(self, input_tensor):
        # First layer
        input_tensor = self.conv1(input_tensor)
        input_tensor = self.relu1(input_tensor)
        input_tensor = self.maxpool1(input_tensor)

        # Second layer
        input_tensor = self.conv2(input_tensor)
        input_tensor = self.relu2(input_tensor)
        input_tensor = self.maxpool2(input_tensor)

        # Flatten the data
        input_tensor = flatten(input_tensor, 1)

        # Fully connected layers
        input_tensor = self.fc1(input_tensor)
        input_tensor = self.relu3(input_tensor)

        # Softmax classifier
        input_tensor = self.fc2(input_tensor)
        return self.logSoftmax(input_tensor)
