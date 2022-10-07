from torch import nn
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

class Residual_Network(nn.Module):
        def __init__(self):
            super(Residual_Network, self).__init__()

            self.cnn_layer1 = nn.Sequential(
                nn.Conv2d(3, 64, 3, 1, 1),
                nn.BatchNorm2d(64),
            )

            self.cnn_layer2 = nn.Sequential(
                nn.Conv2d(64, 64, 3, 1, 1),
                nn.BatchNorm2d(64),
            )

            self.cnn_layer3 = nn.Sequential(
                nn.Conv2d(64, 128, 3, 2, 1),
                nn.BatchNorm2d(128),
            )

            self.cnn_layer4 = nn.Sequential(
                nn.Conv2d(128, 128, 3, 1, 1),
                nn.BatchNorm2d(128),
            )
            self.cnn_layer5 = nn.Sequential(
                nn.Conv2d(128, 256, 3, 2, 1),
                nn.BatchNorm2d(256),
            )
            self.cnn_layer6 = nn.Sequential(
                nn.Conv2d(256, 256, 3, 1, 1),
                nn.BatchNorm2d(256),
            )
            self.fc_layer = nn.Sequential(
                nn.Linear(256* 32* 32, 256),
                nn.ReLU(),
                nn.Linear(256, 11)
            )
            self.relu = nn.ReLU()

        def forward(self, x):
            # input (x): [batch_size, 3, 128, 128]
            # output: [batch_size, 11]

            # Extract features by convolutional layers.
            x1 = self.cnn_layer1(x)

            x1 = self.relu(x1)

            x2 = self.cnn_layer2(x1) + x1

            x2 = self.relu(x2)

            x3 = self.cnn_layer3(x2) 

            x3 = self.relu(x3)

            x4 = self.cnn_layer4(x3) + x3

            x4 = self.relu(x4)

            x5 = self.cnn_layer5(x4) 

            x5 = self.relu(x5)

            x6 = self.cnn_layer6(x5) + x5

            x6 = self.relu(x6)

            # The extracted feature map must be flatten before going to fully-connected layers.
            xout = x6.flatten(1)

            # The features are transformed by fully-connected layers to obtain the final logits.
            xout = self.fc_layer(xout)
            return xout



input = torch.rand(1, 3, 128, 128) 
m = Residual_Network()  
# print(input)
print(m(input))