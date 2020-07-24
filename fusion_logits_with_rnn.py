import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import torch.nn.functional as F
import torch.nn as nn

#######################################################
model_num = 11 # multiple model with different backbone
seq_len = 24 # number of inputs of each subject
class_num = 6 # number of classes for classification tasks
#######################################################
batch_size = 16 # hyper parameter
lstm_layers = 2 # hyper parameter
hidden = 96 # hyper parameter
#######################################################
class ModelStacking(nn.Module):
    def __init__(self):
        super(ModelStacking, self).__init__()
        ratio = 4
        # sequence model
        self.conv_first = nn.Sequential(nn.Conv2d(model_num, 128*ratio, kernel_size=(5, 1), stride=(1,1),padding=(2,0),dilation=1, bias=False),
                                        nn.BatchNorm2d(128*ratio),
                                        nn.ReLU(),
                                        nn.Conv2d(128*ratio, 64*ratio, kernel_size=(3, 1), stride=(1, 1), padding=(2, 0),dilation=2, bias=False),
                                        nn.BatchNorm2d(64*ratio),
                                        nn.ReLU())

        self.conv_res = nn.Sequential(nn.Conv2d(64 * ratio, 64 * ratio, kernel_size=(3, 1), stride=(1, 1),padding=(4, 0),dilation=4, bias=False),
                                        nn.BatchNorm2d(64 * ratio),
                                        nn.ReLU(),
                                        nn.Conv2d(64 * ratio, 64 * ratio, kernel_size=(3, 1), stride=(1, 1),padding=(2, 0),dilation=2, bias=False),
                                        nn.BatchNorm2d(64 * ratio),
                                        nn.ReLU(),)
        # kernel size 3->5
        self.conv_final = nn.Sequential(nn.Conv2d(64*ratio, 1, kernel_size=(5, 1), stride=(1, 1), padding=(1, 0), dilation=1,bias=False))

        # bidirectional GRU
        self.hidden = hidden
        self.lstm = nn.GRU(64*ratio*class_num, self.hidden, num_layers=lstm_layers, batch_first=True, bidirectional=True)
        # kernel size 1->3
        self.final = nn.Sequential(nn.Conv2d(1, class_num, kernel_size=(3, self.hidden*2), stride=(1, 1), padding=(0, 0), dilation=1, bias=True))

        self.linear = nn.Linear(class_num*seq_len,class_num)
    def forward(self, x):
        batch_size, _, _, _ = x.shape

        x = self.conv_first(x)
        x = self.conv_res(x)
        out = self.conv_final(x)
        #################################################
        # bidirectional GRU
        x = x.view(batch_size, 256, -1, class_num)
        x = x.permute(0,2,1,3).contiguous()
        x = x.view(batch_size, x.size()[1], -1).contiguous()
        x, _= self.lstm(x)
        x = x.view(batch_size, 1, -1, self.hidden*2)
        x = self.final(x)
        x = x.permute(0,3,2,1)
        #################################################
        out += x
        out = out.view(batch_size,-1)
        return out
#######################################################
if __name__ == "__main__":
    model = ModelStacking()
    input = torch.rand(batch_size, model_num, 1, class_num)
    output = model(input) # (BATCH_SIZE, NUM_CLASSES)