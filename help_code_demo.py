import csv, torch, os
import enum
import numpy as np
import random

def ACC(mylist):
    tp, fn, fp, tn = mylist[0], mylist[1], mylist[2], mylist[3]
    total = sum(mylist)
    acc = (tp + tn) / total
    return acc


def PPV(mylist):
    tp, fn, fp, tn = mylist[0], mylist[1], mylist[2], mylist[3]
    # for the case: there is no VA segs for the patient, then ppv should be 1
    if tp + fn == 0:
        ppv = 1
    # for the case: there is some VA segs, but the predictions are wrong
    elif tp + fp == 0 and tp + fn != 0:
        ppv = 0
    else:
        ppv = tp / (tp + fp)
    return ppv


def NPV(mylist):
    tp, fn, fp, tn = mylist[0], mylist[1], mylist[2], mylist[3]
    # for the case: there is no non-VA segs for the patient, then npv should be 1
    if tn + fp == 0:
        npv = 1
    # for the case: there is some VA segs, but the predictions are wrong
    elif tn + fn == 0 and tn + fp != 0:
        npv = 0
    else:
        npv = tn / (tn + fn)
    return npv


def Sensitivity(mylist):
    tp, fn, fp, tn = mylist[0], mylist[1], mylist[2], mylist[3]
    # for the case: there is no VA segs for the patient, then sen should be 1
    if tp + fn == 0:
        sensitivity = 1
    else:
        sensitivity = tp / (tp + fn)
    return sensitivity


def Specificity(mylist):
    tp, fn, fp, tn = mylist[0], mylist[1], mylist[2], mylist[3]
    # for the case: there is no non-VA segs for the patient, then spe should be 1
    if tn + fp == 0:
        specificity = 1
    else:
        specificity = tn / (tn + fp)
    return specificity


def BAC(mylist):
    sensitivity = Sensitivity(mylist)
    specificity = Specificity(mylist)
    b_acc = (sensitivity + specificity) / 2
    return b_acc


def F1(mylist):
    precision = PPV(mylist)
    recall = Sensitivity(mylist)
    if precision + recall == 0:
        f1 = 0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)
    return f1


def FB(mylist, beta=2):
    precision = PPV(mylist)
    recall = Sensitivity(mylist)
    if precision + recall == 0:
        f1 = 0
    else:
        f1 = (1+beta**2) * (precision * recall) / ((beta**2)*precision + recall)
    return f1

def stats_report(mylist):
    f1 = round(F1(mylist), 5)
    fb = round(FB(mylist), 5)
    se = round(Sensitivity(mylist), 5)
    sp = round(Specificity(mylist), 5)
    bac = round(BAC(mylist), 5)
    acc = round(ACC(mylist), 5)
    ppv = round(PPV(mylist), 5)
    npv = round(NPV(mylist), 5)

    output = str(mylist) + '\n' + \
             "F-1 = " + str(f1) + '\n' + \
             "F-B = " + str(fb) + '\n' + \
             "SEN = " + str(se) + '\n' + \
             "SPE = " + str(sp) + '\n' + \
             "BAC = " + str(bac) + '\n' + \
             "ACC = " + str(acc) + '\n' + \
             "PPV = " + str(ppv) + '\n' + \
             "NPV = " + str(npv) + '\n'

    print("F-1 = ", F1(mylist))
    print("F-B = ", FB(mylist))
    print("SEN = ", Sensitivity(mylist))
    print("SPE = ", Specificity(mylist))
    print("BAC = ", BAC(mylist))
    print("ACC = ", ACC(mylist))
    print("PPV = ", PPV(mylist))
    print("NPV = ", NPV(mylist))

    return output

def loadCSV(csvf):
    """
    return a dict saving the information of csv
    :param splitFile: csv file name
    :return: {label:[file1, file2 ...]}
    """
    dictLabels = {}
    with open(csvf) as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',')
        next(csvreader, None)  # skip (filename, label)
        for i, row in enumerate(csvreader):
            filename = row[0]
            label = row[1]

            # append filename to current label
            if label in dictLabels.keys():
                dictLabels[label].append(filename)
            else:
                dictLabels[label] = [filename]
    return dictLabels


def txt_to_numpy(filename, row):
    file = open(filename)
    lines = file.readlines()
    datamat = np.arange(row, dtype=np.float)
    row_count = 0
    for line in lines:
        line = line.strip().split(' ')
        datamat[row_count] = line[0]
        row_count += 1

    return datamat


class ToTensor(object):
    def __call__(self, sample):
        text = sample['IEGM_seg']
        return {
            'IEGM_seg': torch.from_numpy(text),
            'label': sample['label']
        }

# demo dataloader
class IEGM_DataSET():
    def __init__(self, root_dir, indice_dir, mode, size, transform=None):
        self.root_dir = root_dir
        self.indice_dir = indice_dir
        self.size = size
        self.names_list = []
        self.transform = transform

        csvdata_all = loadCSV(os.path.join(self.indice_dir, mode + '_indice.csv'))

        for i, (k, v) in enumerate(csvdata_all.items()):
            self.names_list.append(str(k) + ' ' + str(v[0]))

    def __len__(self):
        return len(self.names_list)

    def __getitem__(self, idx):
        text_path = os.path.join(self.root_dir, self.names_list[idx].split(' ')[0])

        if not os.path.isfile(text_path):
            print(text_path + 'does not exist')
            return None

        IEGM_seg = txt_to_numpy(text_path, self.size).reshape(1, self.size, 1)
        label = int(self.names_list[idx].split(' ')[1])
        sample = {'IEGM_seg': IEGM_seg, 'label': label}

        return sample

# meta-learning dataloader
class IEGM_DataSET_meta():
    def __init__(self, root_dir, indice_dir, mode, size, new_features, patientn, sptsampn, qrysampn, transform=None):
        self.root_dir = root_dir
        self.indice_dir = indice_dir
        self.size = size
        self.new_features = new_features
        self.patientn = patientn
        self.sptsampn = sptsampn
        self.qrysampn = qrysampn
        self.VA_dict = {}
        self.NVA_dict = {}
        self.transform = transform

        csvdata_all = loadCSV(os.path.join(self.indice_dir, mode + '_indice.csv'))

        for i, (k, v) in enumerate(csvdata_all.items()):
            patient_tag = str(k).split('-')[0]
            if v[0] == '1':
                if patient_tag in self.VA_dict.keys():
                    self.VA_dict[patient_tag].append(str(k))
                else:
                    self.VA_dict[patient_tag] = [str(k)]
            elif v[0] == '0':
                if patient_tag in self.NVA_dict.keys():
                    self.NVA_dict[patient_tag].append(str(k))
                else:
                    self.NVA_dict[patient_tag] = [str(k)]
                
        diff_patients = self.VA_dict.keys() - self.NVA_dict.keys()
        for patient in diff_patients:
            self.VA_dict.pop(patient)

        diff_patients = self.NVA_dict.keys() - self.VA_dict.keys()
        for patient in diff_patients:
            self.NVA_dict.pop(patient)
        
        pass    # 用于断点测试
    
    def __len__(self):
        return int(1000000)

    def __getitem__(self, idx):
        support_set = []
        query_set = []

        # 2022论文set生成
        if self.new_features:
            selected_patients = random.sample(self.VA_dict.keys(), 2*self.patientn)

            for patient in selected_patients[:self.patientn]:
                va_sublist = random.sample(self.VA_dict[patient], self.sptsampn)
                nva_sublist = random.sample(self.NVA_dict[patient], self.sptsampn)
                support_set.extend(zip(va_sublist, [1]*self.sptsampn))
                support_set.extend(zip(nva_sublist, [0]*self.sptsampn))
            for patient in selected_patients[self.patientn:]:
                va_sublist = random.sample(self.VA_dict[patient], self.qrysampn)
                nva_sublist = random.sample(self.NVA_dict[patient], self.qrysampn)
                query_set.extend(zip(va_sublist, [1]*self.qrysampn))
                query_set.extend(zip(nva_sublist, [0]*self.qrysampn))
        # end of 2022论文set生成
        
        # 2021论文set生成
        else:
            selected_patients = random.sample(self.VA_dict.keys(), self.patientn)
            
            for patient in selected_patients:
                va_sublist = random.sample(self.VA_dict[patient], self.sptsampn+self.qrysampn)
                nva_sublist = random.sample(self.NVA_dict[patient], self.sptsampn+self.qrysampn)
                support_set.extend(zip(va_sublist[:self.sptsampn], [1]*self.sptsampn))
                support_set.extend(zip(nva_sublist[:self.sptsampn], [0]*self.sptsampn))
                query_set.extend(zip(va_sublist[self.sptsampn:], [1]*self.qrysampn))
                query_set.extend(zip(nva_sublist[self.sptsampn:], [0]*self.qrysampn))
        # end of 2021论文set生成

        random.shuffle(support_set)
        random.shuffle(query_set)

        support_filelist, support_label = zip(*support_set)
        query_filelist, query_label = zip(*query_set)

        support_sample = []
        query_sample = []

        for file in support_filelist:
            text_path = os.path.join(self.root_dir, file)
            support_sample.append(txt_to_numpy(text_path, self.size).reshape(1, self.size, 1))
        for file in query_filelist:
            text_path = os.path.join(self.root_dir, file)
            query_sample.append(txt_to_numpy(text_path, self.size).reshape(1, self.size, 1))

        batch = {'support_sample': np.array(support_sample),
                 'support_label': np.array(support_label),
                 'query_sample': np.array(query_sample),
                 'query_label': np.array(query_label)}

        return batch


def pytorch2onnx(net_path, net_name, size):
    net = torch.load(net_path, map_location=torch.device('cpu'))

    dummy_input = torch.randn(1, 1, size, 1)

    optName = str(net_name)+'.onnx'
    torch.onnx.export(net, dummy_input, optName, verbose=True)
