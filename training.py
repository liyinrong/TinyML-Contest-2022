import argparse
from itertools import cycle
from re import L
from sched import scheduler
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from help_code_demo import ToTensor, IEGM_DataSET, IEGM_DataSET_meta
from models.model_fc import IEGMNetFC
from models.model_conv import IEGMNetConv
import sys
import collections
import os

def main():
    # Hyperparameters
    CONTINUE_TRAINING = args.cont_training
    SELECTED_MODEL = args.selected_model
    NEW_FEATURES = args.new_features
    EPOCH_NUM = args.epochn
    LR = args.lr
    MLR = args.mlr
    MLR_UPDATE_HCYCLE = args.mlr_update_hcycle
    MAX_MLR = args.max_mlr
    BATCH_SIZE = args.batchsz
    BENCH_SIZE = args.benchsz
    PATIENT_NUM = args.patientn
    SUPPORT_SAMPLE_NUM = args.sptsampn
    QUERY_SAMPLE_NUM = args.qrysampn
    INNER_LOOP_NUM = args.iloopn
    PRESET_UPDATING_STRIDE = args.psu_stride
    MIN_INNER_LOOP_NUM = args.min_iloopn
    OUTER_LOOP_NUM = args.oloopn
    SIZE = args.size
    path_old_model = args.path_old_model
    path_data = args.path_data
    path_indices = args.path_indices
    path_models = args.path_models

    # Instantiating NN
    if CONTINUE_TRAINING:
        net = torch.load(path_old_model, map_location='cpu')
    elif SELECTED_MODEL == 'conv':
        net = IEGMNetConv()
    elif SELECTED_MODEL == 'fc':
        net = IEGMNetFC()
    else:
        print("Invalid selected model.")
        sys.exit()

    net.train()
    net = net.float().to(device)

    # 测试代码
    # _para = net.parameters()
    # _dict = net.state_dict()
    # bn_param = {}
    # bn_param['conv1.2.running_mean'] = _dict.get('conv1.2.running_mean')
    # bn_param['conv1.2.running_mean'] = torch.tensor([0.5, 0.5, 0.5])
    # net.load_state_dict(bn_param, strict=False)
    # _newdict = _dict = net.state_dict()
    # end of 测试代码

    # Start dataset loading
    trainset = IEGM_DataSET_meta(root_dir=path_data,
                            indice_dir=path_indices,
                            mode='train',
                            size=SIZE,
                            new_features=NEW_FEATURES,
                            patientn=PATIENT_NUM,
                            sptsampn=SUPPORT_SAMPLE_NUM,
                            qrysampn=QUERY_SAMPLE_NUM,
                            transform=transforms.Compose([ToTensor()]))

    trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    benchmarkset = IEGM_DataSET(root_dir=path_data,
                                indice_dir=path_indices,
                                mode='test',
                                size=SIZE,
                                transform=transforms.Compose([ToTensor()]))

    benchmarkloader = DataLoader(benchmarkset, batch_size=BENCH_SIZE, shuffle=True, num_workers=0)

    print("Training Dataset loading finish.")

    inner_loop_num = INNER_LOOP_NUM
    criterion = nn.CrossEntropyLoss()

    if NEW_FEATURES:
        optimizer = optim.SGD(net.parameters(), lr=MLR)
        scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=MLR, max_lr=MAX_MLR, step_size_up=MLR_UPDATE_HCYCLE, mode="triangular2")
    else:
        optimizer = optim.Adam(net.parameters(), lr=MLR)

    # 测试代码
    # for epoch in range(48):
    #     test1 = optimizer.state_dict()
    #     scheduler.step()
    #     pass
    # end of 测试代码

    # 测试代码
    # Bench_acc = []
    # bench_data = next(iter(benchmarkloader))
    # bench_sample, bench_label = bench_data['IEGM_seg'], bench_data['label']
    # bench_sample = bench_sample.float().to(device)
    # bench_label = bench_label.to(device)
    # benchmark_outputs = net(bench_sample)
    # _, bench_pred = torch.max(benchmark_outputs.data, 1)
    # bench_correct = (bench_pred == bench_label).sum()
    # bench_accuracy = bench_correct / float(BENCH_SIZE)
    # Bench_acc.append(bench_accuracy)
    # print('Benchmark Acc: %.5f' % (bench_accuracy))
    # end of 测试代码

    print("Start training")

    if not os.path.exists(path_models):
        os.makedirs(path_models)
    path_best_models = os.path.join(path_models, 'best')
    if not os.path.exists(path_best_models):
        os.makedirs(path_best_models)

    Best_acc = 0.0
    for epoch in range(EPOCH_NUM):  # loop over the dataset multiple times (specify the #epoch)
        Test_acc = []
        Bench_acc = []

        for idx, batch in enumerate(trainloader, 0):
            if idx == OUTER_LOOP_NUM:
                break
            
            inner_loop_num = max(inner_loop_num - idx % PRESET_UPDATING_STRIDE, MIN_INNER_LOOP_NUM)

            batch_loss = []
            batch_bn_params = []
            correct = 0.0
            total = 0.0

            for b in range(BATCH_SIZE):
                support_sample = batch['support_sample'][b]
                support_label = batch['support_label'][b]
                query_sample = batch['query_sample'][b]
                query_label = batch['query_label'][b]

                support_sample = support_sample.float().to(device)
                support_label = support_label.long().to(device)
                query_sample = query_sample.float().to(device)
                query_label = query_label.long().to(device)

                net_params = collections.OrderedDict(net.named_parameters())
                bn_params = net.bn_parameters()
                for n in range(inner_loop_num):
                    training_outputs = net.functional_forward(support_sample, net_params, bn_params)
                    training_loss = criterion(training_outputs, support_label)
                    grads = torch.autograd.grad(training_loss, net_params.values(), create_graph=True)
                    net_params = collections.OrderedDict((name, param - LR * grads)
                                                   for ((name, param), grads) in zip(net_params.items(), grads))

                testing_outputs = net.functional_forward(query_sample, net_params, bn_params)
                testing_loss = criterion(testing_outputs, query_label)

                # 测试代码
                # outer_optimizer.zero_grad()
                # testing_outputs = net(query_sample)
                # testing_loss = outer_criterion(testing_outputs, query_label)
                # testing_loss.backward()
                # outer_optimizer.step()
                # end of 测试代码
                
                batch_loss.append(testing_loss)
                batch_bn_params.append(bn_params)

                _, prediction = torch.max(testing_outputs.data, 1)
                correct += (prediction == query_label).sum()
                total += query_label.size(0)

            optimizer.zero_grad()
            avr_loss = torch.stack(batch_loss).mean()
            avr_loss.backward()
            optimizer.step()
            avr_bn_params = {}
            for subdict in batch_bn_params:
                avr_bn_params.update((k, avr_bn_params.get(k, 0) + v) for k, v in subdict.items())
            avr_bn_params.update((k, v/BATCH_SIZE) for k, v in avr_bn_params.items())
            net.load_bn_parameters(avr_bn_params)

            print('Test Acc: %.5f' % (correct / total))
            Test_acc.append((correct / total).item())
            
            if idx % 50 == 49:
                torch.save(net, os.path.join(path_models, 'meta_' + str(epoch) + '_' + str(idx) + '.pkl'))
                # torch.save(net.state_dict(), './saved_models/IEGM_net_state_dict_' + str(epoch) + '_' + str(idx) + '.pkl')
                net.eval()
                bench_data = next(iter(benchmarkloader))
                bench_sample, bench_label = bench_data['IEGM_seg'], bench_data['label']
                bench_sample = bench_sample.float().to(device)
                bench_label = bench_label.to(device)
                benchmark_outputs = net(bench_sample)
                _, bench_pred = torch.max(benchmark_outputs.data, 1)
                bench_correct = (bench_pred == bench_label).sum()
                bench_accuracy = bench_correct / bench_label.size(0)
                Bench_acc.append(bench_accuracy)
                
                print('Benchmark Acc: %.5f' % (bench_accuracy))
                if bench_accuracy > Best_acc:
                    Best_acc = bench_accuracy
                    torch.save(net, os.path.join(path_best_models, 'meta_' + str(epoch) + '_' + str(idx) + '.pkl'))
                    # torch.save(net.state_dict(), './saved_models/best/IEGM_net_state_dict_' + str(epoch) + '_' + str(idx) + '.pkl')
                net.train()

        Test_acc_file = open(os.path.join(path_models, 'meta_test_acc_' + str(epoch) + '_.txt'), 'w')
        Test_acc_file.write(str(Test_acc))
        Test_acc_file.write('\n\n')
        Test_acc_file.close()

        Bench_acc_file = open(os.path.join(path_models, 'meta_bench_acc.txt'), 'a')
        Bench_acc_file.write('epoch' + str(epoch) + ':\n')
        Bench_acc_file.write(str(Bench_acc))
        Bench_acc_file.write('\n\n')
        Bench_acc_file.close()
        
        if NEW_FEATURES:
            scheduler.step()

    print('Finish training')


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--cont_training', action='store_true', help='continue training on old model', default=False)
    argparser.add_argument('--path_old_model', type=str, help='path of old model for continuous training', default='./model_archive/saved_model.pkl')
    argparser.add_argument('--selected_model', type=str, help='FOR NEW TRAINING ONLY: select fc or conv model', default='conv')
    argparser.add_argument('--new_features', action='store_true', help='activate new task generating mechanism and dynamic learning rate in paper 2022', default=False)
    argparser.add_argument('--epochn', type=int, help='epoch number', default=1)   # 2021：1   2022：50
    argparser.add_argument('--lr', type=float, help='learning rate aka alpha in the paper', default=0.0005)
    argparser.add_argument('--mlr', type=float, help='meta learning rate aka beta in the paper', default=0.0002)    #2021：0.0002   2022：0.0001
    argparser.add_argument('--mlr_update_hcycle', type=int, help='number of epoch required to update mlr for half a cycle, no sense when new_feature is not activated', default=16)
    argparser.add_argument('--max_mlr', type=float, help='max meta-learning rate in cyclical mechanism, no sense when new_features is not activated', default=0.0005)
    argparser.add_argument('--batchsz', type=int, help='number of tasks in a single batch aka B in the paper', default=8)
    argparser.add_argument('--benchsz', type=int, help='size of benchmark', default=5625)
    argparser.add_argument('--patientn', type=int, help='patient number of each set aka N in the paper', default=8)
    argparser.add_argument('--sptsampn', type=int, help='sample number of each patient in support set aka p in the paper', default=4)
    argparser.add_argument('--qrysampn', type=int, help='sample number of each patient in query set aka n-p in the paper', default=4)   #2021：4    2022：6
    argparser.add_argument('--iloopn', type=int, help='initial number of inner-loop update steps aka k in the paper', default=5)
    argparser.add_argument('--psu_stride', type=int, help='preset updating stride aka w in the papaer', default=4)
    argparser.add_argument('--min_iloopn', type=int, help='minimal inner-loop update steps aka K in the paper', default=2)
    argparser.add_argument('--oloopn', type=int, help='number of outer-loops aka batches per epoch', default=5000)   #2021：5000 2022：100
    argparser.add_argument('--device', type=str, help='current device: cuda or cpu', default='cpu')
    argparser.add_argument('--cuda', type=int, help='cuda device id, no sense when current device is cpu', default=0)
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
