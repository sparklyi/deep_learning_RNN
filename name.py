#根据地名分辨国家
import math
import time
import torch
# 绘图
import matplotlib.pyplot as plt
import numpy as np
# 读取数据
import gzip
import csv

from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import Dataset, DataLoader
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# ------------0 parameters-------------#
HIDDEN_SIZE = 100
BATCH_SIZE = 256
N_LAYER = 2
N_EPOCHS = 100
N_CHARS = 128  # 字典长度
USE_GPU = False  # 不用GPU

# ---------------------1 Preparing Data and DataLoad-------------------------------#
class NameDataset(Dataset):
    def __init__(self, is_train_set=True):
        filename = 'names_train.csv' if is_train_set else 'names_test.csv'

        # 访问数据集，使用gzip和csv包
        with open(filename, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter=',')  # 假设用逗号分隔
            rows = list(reader)  # 按行读取（natural，Major）

        self.names = [row[0] for row in rows]
        self.len = len(self.names)
        self.countries = [row[1] for row in rows]
        self.country_list = list(sorted(set(self.countries)))  # set:去除重复，sorted：排序，list：转换为列表
        self.country_dict = self.getCountryDict()
        self.country_num = len(self.country_list)

    def __getitem__(self, index):
        return self.names[index], self.country_dict[self.countries[index]]
        # 取出的names是字符串，country_dict是索引

    def __len__(self):
        return self.len

    def getCountryDict(self):  # Convert list into dictionary.
        country_dict = dict()
        for idx, country_name in enumerate(self.country_list, 0):
            country_dict[country_name] = idx
        return country_dict

    def idx2country(self, index):  # Return country name giving index.
        return self.country_list[index]

    def getCountriesNum(self):  # Return the number of countries.
        return self.country_num


# DataLoade
trainset = NameDataset(is_train_set=True)
trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
testset = NameDataset(is_train_set=False)
testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False)
N_COUNTRY = trainset.getCountriesNum()


# ------------------------------Design  Model-----------------------------------#
def create_tensor(tensor):
    if USE_GPU:
        device = torch.device("cuda:0")
        tensor = tensor.to(device)
    return tensor


class RNNClassifier(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers=1, bidirectional=True):
        super(RNNClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.n_directions = 2 if bidirectional else 1  # bidirectional，双向循环神经网络
        self.embedding = torch.nn.Embedding(input_size, hidden_size)
        self.gru = torch.nn.GRU(hidden_size, hidden_size, n_layers, bidirectional=bidirectional)
        self.fc = torch.nn.Linear(hidden_size * self.n_directions, output_size)

    def _init_hidden(self, batch_size):
        hidden = torch.zeros(self.n_layers * self.n_directions, batch_size, self.hidden_size)
        return create_tensor(hidden)

    def forward(self, input, seq_lengths):
        input = input.t()  # 转置 t -> transpose: input shape : B x S -> S x B
        batch_size = input.size(1)

        hidden = self._init_hidden(batch_size)  # h0
        embedding = self.embedding(input)  # （seqLen,batchSize,hiddenSize)

        # PackedSquence：把为0的填充量去除，把每个样本的长度记录下来，按长度排序后拼接在一起
        gru_input = pack_padded_sequence(embedding, seq_lengths)

        output, hidden = self.gru(gru_input, hidden)
        if self.n_directions == 2:  # 双向循环神经网络有两个hidden
            hidden_cat = torch.cat([hidden[-1], hidden[-2]], dim=1)
        else:
            hidden_cat = hidden[-1]

        fc_output = self.fc(hidden_cat)
        return fc_output


classifier = RNNClassifier(N_CHARS, HIDDEN_SIZE, N_COUNTRY, N_LAYER)

#----------------------3 Construct Loss and Optimizer------------------------------------#
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001)


#-----------------------------------4 Train and Test----------------------------------------------------#
def time_since(since):
    s = time.time() - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def name2list(name):
    arr = [ord(c) for c in name]  # 返回对应字符的 ASCII 数值
    return arr, len(arr)  # 返回元组，列表本身和列表长度


def make_tensors(names, countries):
    sequences_and_lengths = [name2list(name) for name in names]
    name_sequences = [sl[0] for sl in sequences_and_lengths]
    seq_lengths = torch.LongTensor([sl[1] for sl in sequences_and_lengths])
    countries = countries.long()  # countries：国家索引

    # make tensor of name, BatchSize x SeqLen
    seq_tensor = torch.zeros(len(name_sequences), seq_lengths.max()).long()
    for idx, (seq, seq_len) in enumerate(zip(name_sequences, seq_lengths), 0):
        seq_tensor[idx, :seq_len] = torch.LongTensor(seq)
    # 先制作一个全0的tensor，然后将名字贴在上面

    # 排序，sort by length to use pack_padded_sequence
    seq_lengths, perm_idx = seq_lengths.sort(dim=0, descending=True)
    # sort返回两个值，seq_lengths：排完序后的序列（未padding），perm_idx：排完序后对应元素的索引
    seq_tensor = seq_tensor[perm_idx]  # 排序（已padding）
    countries = countries[perm_idx]  # 排序（标签）
    return create_tensor(seq_tensor), create_tensor(seq_lengths), create_tensor(countries)


def trainModel():
    total_loss = 0
    for i, (names, countries) in enumerate(trainloader, 1):
        inputs, seq_lengths, target = make_tensors(names, countries)  # make_tensors
        output = classifier(inputs, seq_lengths.to('cpu'))
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        if i % 10 == 0:
            print(f'[{time_since(start)}] Epoch {epoch} ', end='')
            print(f'[{i * len(inputs)}/{len(trainset)}] ', end='')
            print(f'loss={total_loss / (i * len(inputs))}')
    return total_loss

#test module
def hehe():
    correct = 0
    total = len(testset)
    print("evaluating trained model ...")
    with torch.no_grad():
        for i, (names, countries) in enumerate(testloader, 1):
            inputs, seq_lengths, target = make_tensors(names, countries)  # make_tensors
            output = classifier(inputs, seq_lengths.to('cpu'))
            pred = output.max(dim=1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
        percent = '%.2f' % (100 * correct / total)
        print(f'Test set: Accuracy {correct}/{total} {percent}%')
    return correct / total


if __name__ == '__main__':
    if USE_GPU:
        device = torch.device("cuda:0")
        classifier.to(device)
    start = time.time()
    print("Training for %d epochs..." % N_EPOCHS)
    acc_list = []
    # Train cycle，In every epoch, training and testing the model once.
    for epoch in range(1, N_EPOCHS + 1):
        trainModel()
        acc = hehe()
        acc_list.append(acc)

    # 绘图
    epoch = np.arange(1, len(acc_list) + 1, 1)
    acc_list = np.array(acc_list)
    plt.plot(epoch, acc_list)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid()
    plt.show()