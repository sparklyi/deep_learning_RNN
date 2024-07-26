import math
import time
import torch
# 绘图
import matplotlib.pyplot as plt
import numpy as np
import csv

from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import Dataset, DataLoader
import os
import matplotlib

matplotlib.use("TkAgg")
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# ------------0 parameters-------------#
HIDDEN_SIZE = 100
BATCH_SIZE = 128
N_LAYER = 2
N_EPOCHS = 50
DICT = open("dict.txt", 'r', encoding="UTF-8").read()
dist_list = {}
for k, v in enumerate(DICT):
    dist_list[v] = k
N_CHARS = len(DICT)  # 字典长度
USE_GPU = False  # 不用GPU
learning_rate = 1e-2
TRAIN_NAME = "device_train.csv"
TEST_NAME = "device_test.csv"

# ---------------------1 Preparing Data and DataLoad-------------------------------#
class NameDataset(Dataset):
    def __init__(self, is_train_set=True):
        # filename = 'train_data_bak.csv' if is_train_set else 'train_data_bak.csv'
        # filename = 'device_train.csv' if is_train_set else 'device_test.csv'
        # filename = 'device_train.csv' if is_train_set else 'device_train.csv'
        filename = TRAIN_NAME if is_train_set else TEST_NAME
        # filename = 'train_data_bak2.csv' if is_train_set else 'test_data_bak2.csv'

        # 访问数据集，使用gzip和csv包
        with open(filename, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter=',')  # 假设用逗号分隔
            rows = list(reader)  # 按行读取（natural，Major）

        self.natural = [row[0] for row in rows]
        self.len = len(self.natural)
        self.major = [row[1] for row in rows]
        self.major_list = list(sorted(set(self.major)))  # set:去除重复，sorted：排序，list：转换为列表
        # print(f'self.major_list:{self.major_list}')
        self.major_dict = self.getMajorDict()
        self.major_num = len(self.major_list)

    def __getitem__(self, index):
        return self.natural[index], self.major_dict[self.major[index]]
        # 取出的natural是字符串，major_dict是索引

    def __len__(self):
        return self.len

    def getMajorDict(self):  # Convert list into dictionary.
        major_dict = dict()
        for idx, major_name in enumerate(self.major_list, 0):
            major_dict[major_name] = idx
        return major_dict

    def idx2major(self, index):  # Return major name giving index.
        # print(f"idx {index}:{self.major_list[index]}")
        return self.major_list[index]

    def getMajorNum(self):  # Return the number of Major.
        return self.major_num


# DataLoade
trainset = NameDataset(is_train_set=True)
trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
testset = NameDataset(is_train_set=False)
testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False)
N_major = trainset.getMajorNum()


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


model = RNNClassifier(N_CHARS, HIDDEN_SIZE, N_major, N_LAYER)

# 损失函数和优化器
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


# -----------------------------------4 Train and Test----------------------------------------------------#
def time_since(since):
    s = time.time() - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def natural2list(natural):
    arr = [dist_list[c] for c in natural]
    return arr, len(arr)


def make_tensors(natural, Major):
    sequences_and_lengths = [natural2list(name) for name in natural]
    name_sequences = [sl[0] for sl in sequences_and_lengths]
    seq_lengths = torch.LongTensor([sl[1] for sl in sequences_and_lengths])
    Major = Major.long()  # Major：国家索引

    # make tensor of name, BatchSize x SeqLen
    seq_tensor = torch.zeros(len(name_sequences), seq_lengths.max()).long()
    for idx, (seq, seq_len) in enumerate(zip(name_sequences, seq_lengths), 0):
        seq_tensor[idx, :seq_len] = torch.LongTensor(seq)
    # 先制作一个全0的tensor，然后将名字贴在上面

    # 排序，sort by length to use pack_padded_sequence
    seq_lengths, perm_idx = seq_lengths.sort(dim=0, descending=True)
    # sort返回两个值，seq_lengths：排完序后的序列（未padding），perm_idx：排完序后对应元素的索引
    seq_tensor = seq_tensor[perm_idx]  # 排序（已padding）
    Major = Major[perm_idx]  # 排序（标签）
    # print(f"natural_seq:{name_sequences}")
    return create_tensor(seq_tensor), create_tensor(seq_lengths), create_tensor(Major)


def trainModel():
    total_loss = 0
    for i, (natural, Major) in enumerate(trainloader, 1):
        inputs, seq_lengths, target = make_tensors(natural, Major)  # make_tensors
        output = model(inputs, seq_lengths.to('cpu'))
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # print(i)
        total_loss += loss.item()
        # if i % 10 == 0:
        #     print(f'[{time_since(start)}] Epoch {epoch} ', end='')
        #     print(f'[{i * len(inputs)}/{len(trainset)}] ', end='')
        #     print(f'loss={total_loss / (i * len(inputs))}')
    return total_loss


#test module
def test_loop(epoch):
    correct = 0
    total = len(testset)
    # print(f'total: {total}')
    # print("Evaluating trained model ...")
    with torch.no_grad():
        for i, (natural, Major) in enumerate(testloader, 1):
            # print(f"Major:{Major}")
            inputs, seq_lengths, target = make_tensors(natural, Major)  # make_tensors
            output = model(inputs, seq_lengths.to('cpu'))
            pred = output.max(dim=1, keepdim=True)[1]
            # print(f"target:{target}")
            # print(target.view_as(pred))
            correct += pred.eq(target.view_as(pred)).sum().item()
            # print(f'correct: {correct}')
            # print(f'pred: {pred}')
            # print(pred)

            # for idx in range(len(natural)):
            #     correct_major = trainset.idx2major(target[idx].item())
            #     predicted_major = trainset.idx2major(target[idx].item() if target[idx] == pred[idx] else pred[idx].item())
            #     if target[idx] == pred[idx]:
            #         print(f'目标: {correct_major}, 转换: {predicted_major},{"正确" if target[idx] == pred[idx] else "错误"}')

        percent = '%.2f' % (100 * correct / total)
        print(f'{epoch}/{N_EPOCHS} Test set: Accuracy {correct}/{total} {percent}%')
    return correct / total

def make_tensors_from_input(natural):
    sequence, seq_length = natural2list(natural)
    seq_lengths = torch.LongTensor([seq_length])

    seq_tensor = torch.zeros(1, seq_lengths.max()).long()
    seq_tensor[0, :seq_length] = torch.LongTensor(sequence)

    return seq_tensor, seq_lengths

def predict(natural):
    inputs, seq_lengths = make_tensors_from_input(natural)
    output = model(inputs, seq_lengths.to('cpu'))
    _, predicted_class = output.max(dim=1, keepdim=True)
    predicted_major = trainset.idx2major(predicted_class.item())
    return predicted_major



if __name__ == '__main__':
    sele = input("1.训练模型， 0.测试模型")
    print(sele)
    if sele == "1":
        if USE_GPU:
            device = torch.device("cuda:0")
            model.to(device)
        start = time.time()

        print("Training for %d epochs..." % N_EPOCHS)
        acc_list = []
        # Train cycle，In every epoch, training and testing the model once.
        for epoch in range(1, N_EPOCHS + 1):
            trainModel()
            acc = test_loop(epoch)
            acc_list.append(acc)
        end = time.time()
        print("参数：")
        print(f"batch_size: {BATCH_SIZE}, learning_rate: {learning_rate}, epochs: {N_EPOCHS}")
        print(f"train_size: {len(trainset)}, test_size: {len(testset)}")
        t = end-start
        print(f"总耗时:{t//60}m{int(t)%60}s")
        # 绘图
        epoch = np.arange(1, len(acc_list) + 1, 1)
        acc_list = np.array(acc_list)
        plt.plot(epoch, acc_list)
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.grid()
        plt.show()

        # 保存模型
        torch.save(model.state_dict(), 'model.pth')
    else:
        pass
        # model = RNNClassifier(N_CHARS, HIDDEN_SIZE, N_major, N_LAYER)
        # model.load_state_dict(torch.load('model.pth', weights_only=False))
        # model.eval()
        #
        # while 1:
        #     user_input = input('input:')
        #     if user_input == 'exit':
        #         break
        #     predicted_major = predict(user_input)
        #     print(f"预测答案: {predicted_major}")

