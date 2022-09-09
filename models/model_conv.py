import torch.nn as nn
import torch.nn.functional as F

class IEGMNetConv(nn.Module):
    def __init__(self):
        super(IEGMNetConv, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=3, kernel_size=(6, 1), stride=(2,1), padding=0),
            nn.BatchNorm2d(3, affine=True, track_running_stats=True, eps=1e-5, momentum=0.1),
            nn.ReLU(True),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=5, kernel_size=(5, 1), stride=(2,1), padding=0),
            nn.BatchNorm2d(5, affine=True, track_running_stats=True, eps=1e-5, momentum=0.1),
            nn.ReLU(True),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=5, out_channels=10, kernel_size=(4, 1), stride=(2,1), padding=0),
            nn.BatchNorm2d(10, affine=True, track_running_stats=True, eps=1e-5, momentum=0.1),
            nn.ReLU(True),
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=20, kernel_size=(4, 1), stride=(2,1), padding=0),
            nn.BatchNorm2d(20, affine=True, track_running_stats=True, eps=1e-5, momentum=0.1),
            nn.ReLU(True),
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=20, out_channels=20, kernel_size=(4, 1), stride=(2,1), padding=0),
            nn.BatchNorm2d(20, affine=True, track_running_stats=True, eps=1e-5, momentum=0.1),
            nn.ReLU(True),
        )

        self.conv6 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Conv2d(in_channels=740, out_channels=10, kernel_size=(1, 1), stride=(1,1), padding=0),
            nn.ReLU(True),
        )

        self.conv7 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=2, kernel_size=(1, 1), stride=(1,1), padding=0),
            nn.ReLU(False),
        )


    def forward(self, input):

        conv1_output = self.conv1(input)
        conv2_output = self.conv2(conv1_output)
        conv3_output = self.conv3(conv2_output)
        conv4_output = self.conv4(conv3_output)
        conv5_output = self.conv5(conv4_output)
        # conv5_output = conv5_output.view(-1,740)
        conv5_output = conv5_output.reshape(conv5_output.shape[0],740,1,1)

        conv6_output = self.conv6(conv5_output)
        conv7_output = self.conv7(conv6_output)
        conv7_output = conv7_output.view(-1,2)
        return conv7_output

    def functional_forward(self, input, params, bn_params):
        conv1_output = F.conv2d(input, weight=params[f'conv1.0.weight'], bias=params[f'conv1.0.bias'], stride=(2,1), padding=0)
        conv1_output = F.batch_norm(conv1_output, running_mean=bn_params[f'conv1.1.running_mean'], running_var=bn_params[f'conv1.1.running_var'],
                                    weight=params[f'conv1.1.weight'], bias=params[f'conv1.1.bias'], training=True)
        conv1_output = F.relu(conv1_output, True)

        conv2_output = F.conv2d(conv1_output, weight=params[f'conv2.0.weight'], bias=params[f'conv2.0.bias'], stride=(2,1), padding=0)
        conv2_output = F.batch_norm(conv2_output, running_mean=bn_params[f'conv2.1.running_mean'], running_var=bn_params[f'conv2.1.running_var'],
                                    weight=params[f'conv2.1.weight'], bias=params[f'conv2.1.bias'], training=True)
        conv2_output = F.relu(conv2_output, True)

        conv3_output = F.conv2d(conv2_output, weight=params[f'conv3.0.weight'], bias=params[f'conv3.0.bias'], stride=(2,1), padding=0)
        conv3_output = F.batch_norm(conv3_output, running_mean=bn_params[f'conv3.1.running_mean'], running_var=bn_params[f'conv3.1.running_var'],
                                    weight=params[f'conv3.1.weight'], bias=params[f'conv3.1.bias'], training=True)
        conv3_output = F.relu(conv3_output, True)

        conv4_output = F.conv2d(conv3_output, weight=params[f'conv4.0.weight'], bias=params[f'conv4.0.bias'], stride=(2,1), padding=0)
        conv4_output = F.batch_norm(conv4_output, running_mean=bn_params[f'conv4.1.running_mean'], running_var=bn_params[f'conv4.1.running_var'],
                                    weight=params[f'conv4.1.weight'], bias=params[f'conv4.1.bias'], training=True)
        conv4_output = F.relu(conv4_output, True)

        conv5_output = F.conv2d(conv4_output, weight=params[f'conv5.0.weight'], bias=params[f'conv5.0.bias'], stride=(2,1), padding=0)
        conv5_output = F.batch_norm(conv5_output, running_mean=bn_params[f'conv5.1.running_mean'], running_var=bn_params[f'conv5.1.running_var'],
                                    weight=params[f'conv5.1.weight'], bias=params[f'conv5.1.bias'], training=True)
        conv5_output = F.relu(conv5_output, True)
        # conv5_output = conv5_output.view(-1,740)
        conv5_output = conv5_output.reshape(conv5_output.shape[0],740,1,1)

        conv6_output = F.dropout(conv5_output, 0.5)
        conv6_output = F.conv2d(conv6_output, weight=params[f'conv6.1.weight'], bias=params[f'conv6.1.bias'], stride=(1,1), padding=0)
        conv6_output = F.relu(conv6_output, True)

        conv7_output = F.conv2d(conv6_output, weight=params[f'conv7.0.weight'], bias=params[f'conv7.0.bias'], stride=(1,1), padding=0)
        conv7_output = F.relu(conv7_output, False)
        conv7_output = conv7_output.view(-1,2)

        return conv7_output

    def bn_parameters(self):
        bn_params = {}
        state_dict = self.state_dict()
        bn_params[f'conv1.1.running_mean'] = state_dict[f'conv1.1.running_mean']
        bn_params[f'conv1.1.running_var'] = state_dict[f'conv1.1.running_var']
        bn_params[f'conv2.1.running_mean'] = state_dict[f'conv2.1.running_mean']
        bn_params[f'conv2.1.running_var'] = state_dict[f'conv2.1.running_var']
        bn_params[f'conv3.1.running_mean'] = state_dict[f'conv3.1.running_mean']
        bn_params[f'conv3.1.running_var'] = state_dict[f'conv3.1.running_var']
        bn_params[f'conv4.1.running_mean'] = state_dict[f'conv4.1.running_mean']
        bn_params[f'conv4.1.running_var'] = state_dict[f'conv4.1.running_var']
        bn_params[f'conv5.1.running_mean'] = state_dict[f'conv5.1.running_mean']
        bn_params[f'conv5.1.running_var'] = state_dict[f'conv5.1.running_var']

        return bn_params

    def load_bn_parameters(self, bn_params):
        self.load_state_dict(bn_params, strict=False)