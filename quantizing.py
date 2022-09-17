import argparse
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from help_code_demo import ToTensor, IEGM_DataSET
from models.model_fc import IEGMNetFC
from models.model_conv import IEGMNetConv
from models.model_qat import IEGMNetQ
import os
import sys

def main():
    # Hyperparameters
    dev_name = args.device
    BATCH_SIZE = args.batchsz
    BATCH_SIZE_TEST = args.batchsz
    LR = args.lr
    EPOCH = args.epoch
    SIZE = args.size

    path_meta_model = args.path_meta_model
    path_data = args.path_data
    path_indices = args.path_indices
    path_models = args.path_models

    # Instantiating NN
    if dev_name == 'cpu':
        net_meta = torch.load(path_meta_model, map_location='cpu')
    elif dev_name == 'cuda':
        net_meta = torch.load(path_meta_model, map_location="cuda:" + str(args.cuda))

    net = IEGMNetQ(net_meta)
    net = net.float().to(device)
    net.eval()
    net.fuse_layers()
    net.train()
    # net.qconfig = torch.quantization.qconfig.QConfig(
    #     activation=torch.quantization.observer.MinMaxObserver.with_args(dtype=torch.quint8,quant_min=0, quant_max=255, reduce_range=False),
    #     weight=torch.quantization.observer.MinMaxObserver.with_args(dtype=torch.qint8,quant_min=-128, quant_max=127, qscheme=torch.per_tensor_symmetric))
    net.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    torch.quantization.prepare_qat(net, inplace=True)

    # Start dataset loading
    trainset = IEGM_DataSET(root_dir=path_data,
                            indice_dir=path_indices,
                            mode='train',
                            size=SIZE,
                            transform=transforms.Compose([ToTensor()]))

    trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    testset = IEGM_DataSET(root_dir=path_data,
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
    path_best_models = os.path.join(path_models, 'best')
    if not os.path.exists(path_best_models):
        os.makedirs(path_best_models)

    dummy_input = torch.randn(1, 1, SIZE, 1)
    Best_acc = 0.0
    for epoch in range(epoch_num):  # loop over the dataset multiple times (specify the #epoch)

        running_loss = 0.0
        correct = 0.0
        accuracy = 0.0
        i = 0
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
            break

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
        net_int8 = torch.quantization.convert(net, inplace=False)
        torch.save(net_int8, os.path.join(path_models, 'quan_' + str(epoch) + '.pkl'))
        torch.onnx.export(net_int8, dummy_input, os.path.join(path_models, 'quan_' + str(epoch) + '.onnx'), verbose=True)

        for data_test in testloader:
            IEGM_test, labels_test = data_test['IEGM_seg'], data_test['label']
            IEGM_test = IEGM_test.float().to(device)
            labels_test = labels_test.to(device)
            outputs_test = net_int8(IEGM_test)
            _, predicted_test = torch.max(outputs_test.data, 1)
            total += labels_test.size(0)
            correct += (predicted_test == labels_test).sum()

            loss_test = criterion(outputs_test, labels_test)
            running_loss_test += loss_test.item()
            i += 1

        print('Test Acc: %.5f Test Loss: %.5f' % (correct / total, running_loss_test / i))

        Test_loss.append(running_loss_test / i)
        Test_acc.append((correct / total).item())
        if (correct / total) > Best_acc:
            torch.save(net_int8, os.path.join(path_best_models, 'quan_' + str(epoch) + '.pkl'))
        net.train()

    file = open('./saved_models/quan_loss_acc.txt', 'w')
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
    argparser.add_argument('--path_meta_model', type=str, help='path of meta-learning model for QAT', default='./model_archive/saved_model.pkl')
    argparser.add_argument('--epoch', type=int, help='epoch number', default=7)
    argparser.add_argument('--lr', type=float, help='learning rate', default=0.0001)
    argparser.add_argument('--batchsz', type=int, help='total batchsz for traindb', default=32)
    argparser.add_argument('--device', type=str, help='current device: cuda or cpu', default='cpu')
    argparser.add_argument('--cuda', type=int, default=0)
    argparser.add_argument('--size', type=int, default=1250)
    argparser.add_argument('--path_data', type=str, default='./tinyml_contest_data_training')
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
