import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import torch.nn.functional as F
import torch.nn as nn

#######################################################
NUM_CLASSES = 2 # number of classes for classification tasks
NUM_MODELS = 2 # multiple model with different backbone
NUM_CHANNELS = 128 # hyper parameter
BATCH_SIZE = 16
#######################################################
class ModelStacking(nn.Module):
    def __init__(self, num_models, num_channels):
        super(ModelStacking, self).__init__()
        self.conv1 =  nn.Conv2d(1, num_channels,kernel_size=(num_models, 1))
        self.relu1 = nn.ReLU(inplace=True)
        self.dp1 = nn.Dropout(0.3)
        self.conv2 = nn.Conv2d(num_channels, num_channels * 2, kernel_size=(1, NUM_CLASSES))
        self.relu2 = nn.ReLU(inplace=True)
        self.dp2 = nn.Dropout(0.3)
        self.linear = nn.Linear(num_channels * 2, NUM_CLASSES)
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.dp1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.dp2(x)
        x = x.view(x.size()[0],-1)
        x = self.linear(x)
        return x
#######################################################
if __name__ == "__main__":
    model = ModelStacking(NUM_MODELS, NUM_CHANNELS)
    input = torch.rand(BATCH_SIZE, 1, NUM_MODELS, NUM_CLASSES)
    output = model(a) # (BATCH_SIZE, NUM_CLASSES)