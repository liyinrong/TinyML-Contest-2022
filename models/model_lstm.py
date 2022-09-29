import torch.nn as nn
import torch.nn.functional as F

class IEGMNetLSTM(nn.Module):
    def __init__(self):
        super(IEGMNetLSTM, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=128, kernel_size=20, stride=3, padding=0),
            nn.BatchNorm1d(128, affine=True, track_running_stats=True, eps=1e-5, momentum=0.1),
            nn.ReLU(True),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=32, kernel_size=7, stride=1, padding=0),
            nn.BatchNorm1d(32, affine=True, track_running_stats=True, eps=1e-5, momentum=0.1),
            nn.ReLU(True),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=10, stride=1, padding=0),
            nn.BatchNorm1d(32, affine=True, track_running_stats=True, eps=1e-5, momentum=0.1),
            nn.ReLU(True),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        self.lstm = nn.LSTM(input_size=32, hidden_size=10, batch_first=True)

        self.flat = nn.Flatten()

        self.fc1 = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(in_features=450, out_features=10),
            nn.ReLU(True)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(in_features=10, out_features=2),
            nn.ReLU(True)
        )

    def forward(self, input):

        conv1_output = self.conv1(input)
        conv2_output = self.conv2(conv1_output)
        conv3_output = self.conv3(conv2_output)

        conv3_output = conv3_output.transpose(1,2)
        lstm_output, _ = self.lstm(conv3_output)
        flat_output = self.flat(lstm_output)

        fc1_output = self.fc1(flat_output)
        fc2_output = self.fc2(fc1_output)
        return fc2_output