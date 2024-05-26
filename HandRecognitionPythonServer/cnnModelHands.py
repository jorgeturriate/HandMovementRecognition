import torch
import torch.nn as nn
import torch.nn.functional as F



class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction_ratio=8):
        super(SEBlock, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1)
        self.fc2 = nn.Conv2d(in_channels // reduction_ratio, in_channels, kernel_size=1)

    def forward(self, x):
        y = self.pool(x)
        y = F.relu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y))
        return x * y

class SEBlockWithConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(SEBlockWithConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        self.activation = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.se = SEBlock(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        x = self.pool(x)
        x = self.se(x)
        return x

class RecognitionNetwork(nn.Module):
    def __init__(self, input_channels=3, num_classes=10):
        super(RecognitionNetwork, self).__init__()
        self.se_block1_1 = SEBlockWithConv(input_channels, 32, kernel_size=(3, 7), stride=(1, 1))
        self.se_block1_2 = SEBlockWithConv(32, 64, kernel_size=(3, 5), stride=(1, 1))
        self.se_block1_3 = SEBlockWithConv(64, 128, kernel_size=(3, 3), stride=(1, 1))
        self.fc_spatial1 = nn.Linear(128, 128)
        self.dropout_spatial1 = nn.Dropout(p=0.5)
        self.fc_spatial2 = nn.Linear(128, 256)
        self.dropout_spatial2 = nn.Dropout(p=0.5)

        self.se_block2_1 = SEBlockWithConv(input_channels, 32, kernel_size=(3, 7), stride=(1, 1))
        self.se_block2_2 = SEBlockWithConv(32, 64, kernel_size=(3, 5), stride=(1, 1))
        self.se_block2_3 = SEBlockWithConv(64, 128, kernel_size=(3, 3), stride=(1, 1))
        self.fc_temporal1 = nn.Linear(128, 128)
        self.dropout_temporal1 = nn.Dropout(p=0.5)
        self.fc_temporal2 = nn.Linear(128, 512)
        self.dropout_temporal2 = nn.Dropout(p=0.5)

        self.fc_fusion1 = nn.Linear(768, 256)
        self.dropout_fusion1 = nn.Dropout(p=0.5)

        self.fc_classification1 = nn.Linear(256, num_classes)

    def forward(self, spatial_input, temporal_input):
        x1 = self.se_block1_1(spatial_input)
        x1 = self.se_block1_2(x1)
        x1 = self.se_block1_3(x1)
        x1 = F.avg_pool2d(x1, x1.size()[2:]).view(x1.size()[0], -1)
        x1 = F.relu(self.fc_spatial1(x1))
        x1 = self.dropout_spatial1(x1)
        x1 = F.relu(self.fc_spatial2(x1))
        x1 = self.dropout_spatial2(x1)

        x2 = self.se_block2_1(temporal_input)
        x2 = self.se_block2_2(x2)
        x2 = self.se_block2_3(x2)
        x2 = F.avg_pool2d(x2, x2.size()[2:]).view(x2.size()[0], -1)
        x2 = F.relu(self.fc_temporal1(x2))
        x2 = self.dropout_temporal1(x2)
        x2 = F.relu(self.fc_temporal2(x2))
        x2 = self.dropout_temporal2(x2)

        x = torch.cat((x1, x2), dim=1)
        x = F.relu(self.fc_fusion1(x))
        x = self.dropout_fusion1(x)

        x = self.fc_classification1(x)
        return x


model = RecognitionNetwork(input_channels=3, num_classes=10)
model.load_state_dict(torch.load('northwestern_classifier47.pt', map_location=torch.device('cpu')))
model.eval()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)