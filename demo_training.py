import argparse
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from help_code_demo import ToTensor, IEGM_DataSET, IEGM_DataSET_1D
from models.model_fc import IEGMNetFC
from models.model_conv import IEGMNetConv
from models.model_lstm import IEGMNetLSTM
import sys
import os
import collections

def main():
    # Hyperparameters
    SELECTED_MODEL = args.selected_model
    BATCH_SIZE = args.batchsz
    BATCH_SIZE_TEST = args.batchsz
    LR = args.lr
    EPOCH = args.epoch
    SIZE = args.size
    path_data = args.path_data
    path_indices = args.path_indices
    path_models = args.path_models

    # Instantiating NN
    if SELECTED_MODEL == 'conv':
        net = IEGMNetConv()
    elif SELECTED_MODEL == 'fc':
        net = IEGMNetFC()
    elif SELECTED_MODEL == 'lstm':
        net = IEGMNetLSTM()
    else:
        print("Invalid selected model.")
        sys.exit()

    net.train()
    net = net.float().to(device)
    ttt = collections.OrderedDict(net.named_parameters())
    # torch.save(net, 'test.pkl')
    # dummy_input = torch.randn(1, 1, 1250)
    # test_output = net(dummy_input)

    # Start dataset loading
    trainset = IEGM_DataSET_1D(root_dir=path_data,
                            indice_dir=path_indices,
                            mode='train',
                            size=SIZE,
                            transform=transforms.Compose([ToTensor()]))

    trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    testset = IEGM_DataSET_1D(root_dir=path_data,
                           indice_dir=path_indices,
                           mode='test',
                           size=SIZE,
                           transform=transforms.Compose([ToTensor()]))

    testloader = DataLoader(testset, batch_size=BATCH_SIZE_TEST, shuffle=True, num_workers=0)

    print("Training Dataset loading finish.")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=LR)
    epoch_num = EPOCH

    Train_loss = []
    Train_acc = []
    Test_loss = []
    Test_acc = []

    print("Start training")

    if not os.path.exists(path_models):
        os.makedirs(path_models)

    for epoch in range(epoch_num):  # loop over the dataset multiple times (specify the #epoch)
        running_loss = 0.0
        correct = 0.0
        accuracy = 0.0
        i = 0
        net.train()
        for j, data in enumerate(trainloader, 0):
            inputs, labels = data['IEGM_seg'], data['label']
            inputs = inputs.float().to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum()
            accuracy += correct / BATCH_SIZE
            correct = 0.0

            running_loss += loss.item()
            i += 1

        print('[Epoch, Batches] is [%d, %5d] \nTrain Acc: %.5f Train loss: %.5f' %
              (epoch + 1, i, accuracy / i, running_loss / i))

        Train_loss.append(running_loss / i)
        Train_acc.append((accuracy / i).item())

        running_loss = 0.0
        accuracy = 0.0

        correct = 0.0
        total = 0.0
        i = 0.0
        running_loss_test = 0.0
        net.eval()
        for data_test in testloader:
            IEGM_test, labels_test = data_test['IEGM_seg'], data_test['label']
            IEGM_test = IEGM_test.float().to(device)
            labels_test = labels_test.to(device)
            outputs_test = net(IEGM_test)
            _, predicted_test = torch.max(outputs_test.data, 1)
            total += labels_test.size(0)
            correct += (predicted_test == labels_test).sum()

            loss_test = criterion(outputs_test, labels_test)
            running_loss_test += loss_test.item()
            i += 1

        print('Test Acc: %.5f Test Loss: %.5f' % (correct / total, running_loss_test / i))

        Test_loss.append(running_loss_test / i)
        Test_acc.append((correct / total).item())

        torch.save(net, os.path.join(path_models, str(epoch) + '.pkl'))
        # torch.save(net.state_dict(), './saved_models/IEGM_net_state_dict.pkl')

    file = open('./saved_models/loss_acc.txt', 'w')
    file.write("Train_loss\n")
    file.write(str(Train_loss))
    file.write('\n\n')
    file.write("Train_acc\n")
    file.write(str(Train_acc))
    file.write('\n\n')
    file.write("Test_loss\n")
    file.write(str(Test_loss))
    file.write('\n\n')
    file.write("Test_acc\n")
    file.write(str(Test_acc))
    file.write('\n\n')

    print('Finish training')


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--selected_model', type=str, help='FOR NEW TRAINING ONLY: select fc or conv model', default='lstm')
    argparser.add_argument('--epoch', type=int, help='epoch number', default=50)
    argparser.add_argument('--lr', type=float, help='learning rate', default=0.0002)
    argparser.add_argument('--batchsz', type=int, help='total batchsz for traindb', default=32)
    argparser.add_argument('--device', type=str, help='current device: cuda or cpu', default='cpu')
    argparser.add_argument('--cuda', type=int, default=0)
    argparser.add_argument('--size', type=int, default=1250)
    argparser.add_argument('--path_data', type=str, default='./tinyml_contest_data_training/')
    argparser.add_argument('--path_indices', type=str, default='./data_indices')
    argparser.add_argument('--path_models', type=str, default='./saved_models')

    args = argparser.parse_args()

    dev_name = args.device
    if dev_name == 'cpu':
        device = torch.device('cpu')
    elif dev_name == 'cuda':
        device = torch.device("cuda:" + str(args.cuda))
    else:
        print("Invalid current device.")
        sys.exit()

    print("device is --------------", device)

    main()
