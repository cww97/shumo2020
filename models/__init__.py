import torch
import torch.nn as nn
import torch.nn.functional as F


class EEGNet(nn.Module):
    '''
    https://arxiv.org/abs/1611.08024
    '''
    def __init__(self, dropout=0):
        super(EEGNet, self).__init__()
        self.dropout_rate = dropout
        
        # Layer 1
        self.conv1 = nn.Conv2d(1, 16, (20, 1), padding = 0)  # TODO
        self.batchnorm1 = nn.BatchNorm2d(16, False)
        
        # Layer 2
        self.padding1 = nn.ZeroPad2d((9, 10, 0, 1))
        self.conv2 = nn.Conv2d(1, 4, (2, 20))
        self.batchnorm2 = nn.BatchNorm2d(4, False)
        self.pooling2 = nn.MaxPool2d(2, 4)
        
        # Layer 3
        self.padding2 = nn.ZeroPad2d((2, 1, 4, 3))
        self.conv3 = nn.Conv2d(4, 4, (8, 4))  # change? TODO
        self.batchnorm3 = nn.BatchNorm2d(4, False)
        self.pooling3 = nn.MaxPool2d((2, 4))
        
        # FC Layer
        # NOTE: This dimension will depend on the number of timestamps per sample in your data.
        # I have 120 timepoints.
        self.fc1 = nn.Linear(16, 2)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.normal(m.weight.data)
                # nn.init.xavier_normal(m.weight.data)
                nn.init.kaiming_normal(m.weight.data)
                m.bias.data.fill_(0)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_()

    def forward(self, x):

        # Layer 1
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = F.elu(x)
        x = F.dropout(x, self.dropout_rate)
        x = x.permute(0, 2, 1, 3)
        
        # Layer 2
        x = self.padding1(x)  # same TODO
        x = self.conv2(x)
        # x = self.batchnorm2(x)
        x = F.elu(x)
        x = F.dropout(x, self.dropout_rate)
        x = self.pooling2(x)
        
        # Layer 3
        x = self.padding2(x)  # same TODO
        x = self.conv3(x)
        # x = self.batchnorm3(x)
        x = F.elu(x)
        x = F.dropout(x, self.dropout_rate)
        x = self.pooling3(x)
        
        # FC Layer
        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        # import pdb; pdb.set_trace()
        # x = F.elu(x)
        x = self.relu(x)
        # x = self.sigmoid(x)
        x = self.softmax(x)
        return x
