import os
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from numpy.lib.stride_tricks import sliding_window_view
from torch.utils.data import Dataset, DataLoader


def load_adj():
    dataset_path = '../BjTT'
    # dataset_path = os.path.join(dataset_path, dataset_name)
    # adj = sp.load_npz(os.path.join(dataset_path, 'adj.npz'))
    # adj = adj.tocsc()
    adj = np.load(dataset_path + '/' + 'matrix.npy')

    # if dataset_name == 'metr-la':
    #     n_vertex = 207
    # elif dataset_name == 'pems-bay':
    #     n_vertex = 325
    # elif dataset_name == 'pemsd7-m':
    #     n_vertex = 228
    n_vertex = 1260

    return adj, n_vertex

class TrafficTextDataset(Dataset):
    def __init__(self, numeric_data, text_data, labels):
        """
        numeric_data: (N_samples, T, N, F_num)
        text_data:    (N_samples, T, 768)   # BERT embeddings
        labels:       (N_samples, T_pred, N, F_out)
        """
        self.numeric = numeric_data
        self.text = text_data
        self.labels = labels

    def __len__(self):
        return len(self.numeric)

    def __getitem__(self, idx):
        x_num = self.numeric[idx]
        x_text = self.text[idx]
        y = self.labels[idx]
        return x_num, x_text, y


# ========== Step 3: 滑动窗口处理 ==========
def window_text_embeddings(file_list, data_folder,hidden_size, window_size=30):
    """
    输入：所有文本的 embedding [num_samples, hidden_size]
    输出：滑动窗口后的 [num_windows, window_size, hidden_size]
    """
    file_list = [f.decode('utf-8') if isinstance(f, bytes) else f for f in file_list]
    num_files = len(file_list)

    # 提前分配最终大数组: (num_files, 1260, 3)
    # float32 节省内存（比 float64 节省一半）
    data = np.zeros((num_files, hidden_size), dtype=np.float32)

    for i, file_name in enumerate(file_list):
        file_path = os.path.join(data_folder, file_name + ".npy")

        # 用 mmap_mode 避免一次性把整个文件读入内存
        arr = np.load(file_path, mmap_mode='r')  # shape: (1260, 2) 或更大
        data[i, :] = arr

    # 用 sliding_window_view 生成滑窗视图（零拷贝）
    # shape: (num_samples, 1, window_size, hidden_size)
    output = sliding_window_view(data, (window_size, hidden_size))
    # print("embdding shape", output.shape)

    return output


# ========== Step 4: 结合 train/test 划分 ==========
def load_text_data(mode, train_files,val_files, test_files, hidden_size, window_size=30):
    dataset_path = '../BjTT/'
    window_size = 30
    with open(train_files, 'rb') as f:
        train_names = [line.strip() for line in f if line.strip()]  # 去除空行和换行符
    with open(val_files, 'rb') as f:
        val_names = [line.strip() for line in f if line.strip()]  # 去除空行和换行符
    with open(test_files, 'rb') as f:
        test_names = [line.strip() for line in f if line.strip()]  # 去除空行和换行符


    num_test1 = len(test_names[:1080]) - window_size + 1
    num_test2 = len(test_names[1080:]) - window_size + 1
    test = np.zeros((num_test1 + num_test2, window_size, hidden_size), dtype=np.float32)

    if "test" in mode:
        train = window_text_embeddings(train_names, dataset_path, hidden_size, window_size).squeeze(1)
    else:
        # num_train1 = len(train_names[:21240]) - window_size + 1
        # num_train2 = len(train_names[21240:]) - window_size + 1
        num_train1 = len(train_names[:14760]) - window_size + 1
        num_train2 = len(train_names[14760:]) - window_size + 1
        train = np.zeros((num_train1 + num_train2, window_size, hidden_size), dtype=np.float32)

        train1 = window_text_embeddings(train_names[:14760], dataset_path, hidden_size, window_size)
        train[:num_train1] = train1.squeeze(1)
        train2 = window_text_embeddings(train_names[14760:], dataset_path, hidden_size, window_size)
        train[num_train1:] = train2.squeeze(1)


    test1 = window_text_embeddings(test_names[:1080], dataset_path, hidden_size, window_size)
    test[:num_test1] = test1.squeeze(1)
    test2 = window_text_embeddings(test_names[1080:], dataset_path, hidden_size, window_size)
    test[num_test1:] = test2.squeeze(1)
    val = window_text_embeddings(val_names, dataset_path, hidden_size, window_size).squeeze(1)
    # print(train.shape, val.shape, test.shape)

    return train, val, test


def load_data(mode, train_files, val_files, test_files, F_num):
    dataset_path = '../BjTT/'
    window_size = 30
    # dataset_path = os.path.join(dataset_path, dataset_name)
    # vel = pd.read_csv(os.path.join(dataset_path, 'vel.csv'))

    # train = vel[: len_train]
    # val = vel[len_train: len_train + len_val]
    # test = vel[len_train + len_val:]
    # train_data = os.path.join(dataset_path, train_files)
    # test_data = os.path.join(dataset_path, test_files)
    with open(train_files, 'rb') as f:
        train_names = [line.strip() for line in f if line.strip()]  # 去除空行和换行符
    with open(val_files, 'rb') as f:
        val_names = [line.strip() for line in f if line.strip()]  # 去除空行和换行符

    with open(test_files, 'rb') as f:
        test_names = [line.strip() for line in f if line.strip()]  # 去除空行和换行符


    num_test1 = len(test_names[:1080]) - window_size + 1
    num_test2 = len(test_names[1080:]) - window_size + 1

    N = 1260
    print("Start loading data by window size...")
    test = np.zeros((num_test1 + num_test2, window_size, N, F_num), dtype=np.float32)

    if "test" in mode:
        train = load_and_window(train_names, dataset_path, F_num, window_size)
    else:
        # num_train1 = len(train_names[:21240]) - window_size + 1
        # num_train2 = len(train_names[21240:]) - window_size + 1
        num_train1 = len(train_names[:14760]) - window_size + 1
        num_train2 = len(train_names[14760:]) - window_size + 1
        train = np.zeros((num_train1 + num_train2, window_size, N, F_num), dtype=np.float32)

        train1 = load_and_window(train_names[:14760], dataset_path, F_num, window_size)
        train[:num_train1] = train1
        train2 = load_and_window(train_names[14760:], dataset_path, F_num, window_size)
        train[num_train1:] =train2

    test1 = load_and_window(test_names[:1080], dataset_path, F_num, window_size)
    test[:num_test1] = test1
    test2 = load_and_window(test_names[1080:], dataset_path, F_num,  window_size)
    test[num_test1:] = test2
    val = load_and_window(val_names, dataset_path, F_num, window_size)

    return train, val, test


def load_and_window(file_list, data_folder,F_num, window_size=30):
    """
    读取多个 .npy 文件，原始数据列：
    [congestion_level, speed]  ->  输出: [congestion_level, speed, travel_time]

    输出 shape: [num_samples, window_size, 1260, 3]
    """
    file_list = [f.decode('utf-8') if isinstance(f, bytes) else f for f in file_list]
    num_files = len(file_list)

    data = np.zeros((num_files, 1260, F_num), dtype=np.float32)

    for i, file_name in enumerate(file_list):
        file_path = os.path.join(data_folder, file_name + ".npy")

        # 用 mmap_mode 避免一次性把整个文件读入内存
        arr = np.load(file_path, mmap_mode='r')  # shape: (1260, 2)
        arr = arr[:,:35,:].reshape(-1, 2)
        # congestion level
        data[i, :, 0] = arr[:, 0]
        if F_num == 2 or F_num == 3:
            data[i, :, 1] = arr[:, 1]
        # elif F_num == 3:
        #     # travel time calculation
        #     speed_safe = np.maximum(arr[:, 1], 1e-3)  # 避免除 0
        #     travel_time = 3600.0 / speed_safe
        #     travel_time[arr[:, 1] <= 0.0] = 0.0  # speed=0 -> travel_time=0
        #     data[i, :, 2] = travel_time

    # print(data.shape)   ## b,N,F
    output = sliding_window_view(data, window_shape=window_size, axis=0)
    output = np.moveaxis(output, -1, 1)
    # print(output.shape) ## b, T,F, windowsize

    return output

def data_transform(data, n_his, n_pred):
    # produce data slices for x_data and y_data

    # n_vertex = data.shape[1]

    # len_record = len(data)
    # num = len_record - n_his - n_pred
    # x = np.zeros([num, 1, n_his, n_vertex])
    # y = np.zeros([num, n_vertex])
    #
    # for i in range(num):
    #     head = i
    #     tail = i + n_his
    #     x[i, :, :, :] = data[head: tail].reshape(1, n_his, n_vertex)
    #     y[i] = data[tail + n_pred - 1]

    x = data[:,:n_his]
    # x = np.expand_dims(x, axis=1)
    y = data[:,n_his:n_pred+n_his]
    # y = np.expand_dims(y, axis=1)
    # print(x.shape,y.shape)

    return torch.from_numpy(x), torch.from_numpy(y)
