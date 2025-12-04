import os # 操作系统接口交互
import ujson   # json格式数据解析
import numpy as np
import gc
from sklearn.model_selection import train_test_split   # 训练集测试集划分
from torch.utils.data import Dataset   # 数据集基类
from PIL import Image   # 图像处理库


batch_size = 10
train_ratio = 0.75 # merge original training set and test set, then split it manually. 
alpha = 0.1 # for Dirichlet distribution. 100 for exdir
# alpha是Dirichlet分布的超参数，用来控制每个客户端的分布。

def check(config_path, train_path, test_path, num_clients, niid=False, 
        balance=True, partition=None):
    # check existing dataset
    if os.path.exists(config_path):       # 如果已经生成过数据集，则直接返回True，检查配置文件，通常是一个json文件
        with open(config_path, 'r') as f:    #   存在就读取json文件
            config = ujson.load(f)
        if config['num_clients'] == num_clients and \
            config['non_iid'] == niid and \
            config['balance'] == balance and \
            config['partition'] == partition and \
            config['alpha'] == alpha and \
            config['batch_size'] == batch_size:
            print("\nDataset already generated.\n")
            return True
# 上面是检查参数是否一致，如果一致则直接返回True，不再生成数据集。

    dir_path = os.path.dirname(train_path)  #为即将生成的json文件和训练集、测试集创建父级目录
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    dir_path = os.path.dirname(test_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    return False

# 以下函数用于将数据集划分为多个客户端，每个客户端的训练集、测试集、类别数、数据量等信息。
def separate_data(data, num_clients, num_classes, niid=False, balance=False, partition=None, class_per_client=None):
    X = [[] for _ in range(num_clients)]
    y = [[] for _ in range(num_clients)]
    statistic = [[] for _ in range(num_clients)]
#初始化三个列表，又是老生常谈的x,y,statistic了

    dataset_content, dataset_label = data    # data是元组，传给了他数据集和标签数据 拆分数据集和标签
    # guarantee that each client must have at least one batch of data for testing. 
    least_samples = int(min(batch_size / (1-train_ratio), len(dataset_label) / num_clients / 2))
#计算每个客户端至少需要多少样本，第一个是限制至少需要一个batch_size，要满足batch_size=10，必须10/0.25=40个，第二个就是平均分配了，取这两个的最小值
    dataidx_map = {}
# 初始化一个字典，用于保存每个客户端的样本索引
    if not niid:
        partition = 'pat'                # 若不是非IID，则默认用'pat'划分数据集
        class_per_client = num_classes   # 每个客户端拥有所有的类别数

    if partition == 'pat':
        idxs = np.array(range(len(dataset_label)))    # 创建包含所有样本的索引的数组
        idx_for_each_class = []
        for i in range(num_classes):                  # 遍历每个类别，将其索引保存到列表中，idx_for_each_class[0] 存放了所有“数字0”的索引，依此类推
            idx_for_each_class.append(idxs[dataset_label == i])

        class_num_per_client = [class_per_client for _ in range(num_clients)]     # 每个客户端拥有的类别数，初始化为class_per_client
        for i in range(num_classes):                          # 遍历每个类别，准备分配给每个客户端
            selected_clients = []                             # 选择的客户端列表
            for client in range(num_clients):                 # 遍历每个客户端
                if class_num_per_client[client] > 0:          # 若客户端拥有剩余的类别数，则选择该客户端
                    selected_clients.append(client)           # 选择该客户端
            if len(selected_clients) == 0:                     # 若没有可选择的客户端，则退出循环
                break
            selected_clients = selected_clients[:int(np.ceil((num_clients/num_classes)*class_per_client))]   # 这段代码是定义了每个客户端分配的类别数，如果balance=True，则每个客户端分配的类别数相同，否则随机分配

            num_all_samples = len(idx_for_each_class[i])        # 计算每个类别中所有样本的数量和选择的客户端数量
            num_selected_clients = len(selected_clients)
            num_per = num_all_samples / num_selected_clients
            if balance:                                         # 根据是否平衡确定每个客户端分配的类别数
                num_samples = [int(num_per) for _ in range(num_selected_clients-1)]
            else:
                num_samples = np.random.randint(max(num_per/10, least_samples/num_classes), num_per, num_selected_clients-1).tolist()
            num_samples.append(num_all_samples-sum(num_samples))

            idx = 0
            for client, num_sample in zip(selected_clients, num_samples):
                if client not in dataidx_map.keys():
                    dataidx_map[client] = idx_for_each_class[i][idx:idx+num_sample]
                else:
                    dataidx_map[client] = np.append(dataidx_map[client], idx_for_each_class[i][idx:idx+num_sample], axis=0)
                idx += num_sample
                class_num_per_client[client] -= 1
            # 根据计算好的数量 num_sample，从 idx_for_each_class[i] 中切片，把索引塞给客户端。


    elif partition == "dir":            # 若是dir，则分配数据集给客户端，每个客户端的类别数相同，但每个客户端的样本数不同。
        # https://github.com/IBM/probabilistic-federated-neural-matching/blob/master/experiment.py
        min_size = 0
        K = num_classes
        N = len(dataset_label)

        try_cnt = 1                 # 初始化尝试次数计数器，如果分配的客户端数据大小不满足最小要求，则打印信息并增加尝试次数。
        while min_size < least_samples:
            if try_cnt > 1:
                print(f'Client data size does not meet the minimum requirement {least_samples}. Try allocating again for the {try_cnt}-th time.')

            idx_batch = [[] for _ in range(num_clients)]
            for k in range(K):        # 遍历每个类别，开始分配数据集给客户端
                idx_k = np.where(dataset_label == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
                proportions = np.array([p*(len(idx_j)<N/num_clients) for p,idx_j in zip(proportions,idx_batch)])
                proportions = proportions/proportions.sum()
                proportions = (np.cumsum(proportions)*len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j,idx in zip(idx_batch,np.split(idx_k,proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])
            try_cnt += 1

        for j in range(num_clients):
            dataidx_map[j] = idx_batch[j]         # 将分配的样本索引保存到dataidx_map字典中
    
    elif partition == 'exdir':
        r'''This strategy comes from https://arxiv.org/abs/2311.03154
        See details in https://github.com/TsingZ0/PFLlib/issues/139

        This version in PFLlib is slightly different from the original version 
        Some changes are as follows:
        n_nets -> num_clients, n_class -> num_classes
        '''
        C = class_per_client
        
        '''The first level: allocate labels to clients
        clientidx_map (dict, {label: clientidx}), e.g., C=2, num_clients=5, num_classes=10
            {0: [0, 1], 1: [1, 2], 2: [2, 3], 3: [3, 4], 4: [4, 5], 5: [5, 6], 6: [6, 7], 7: [7, 8], 8: [8, 9], 9: [9, 0]}
        '''
        min_size_per_label = 0
        # You can adjust the `min_require_size_per_label` to meet you requirements
        min_require_size_per_label = max(C * num_clients // num_classes // 2, 1)
        if min_require_size_per_label < 1:
            raise ValueError
        clientidx_map = {}
        while min_size_per_label < min_require_size_per_label:
            # initialize
            for k in range(num_classes):
                clientidx_map[k] = []
            # allocate
            for i in range(num_clients):
                labelidx = np.random.choice(range(num_classes), C, replace=False)
                for k in labelidx:
                    clientidx_map[k].append(i)
            min_size_per_label = min([len(clientidx_map[k]) for k in range(num_classes)])
        
        '''The second level: allocate data idx'''
        dataidx_map = {}
        y_train = dataset_label
        min_size = 0
        min_require_size = 10
        K = num_classes
        N = len(y_train)
        print("\n*****clientidx_map*****")
        print(clientidx_map)
        print("\n*****Number of clients per label*****")
        print([len(clientidx_map[i]) for i in range(len(clientidx_map))])

        # ensure per client' sampling size >= min_require_size (is set to 10 originally in [3])
        while min_size < min_require_size:
            idx_batch = [[] for _ in range(num_clients)]
            # for each class in the dataset
            for k in range(K):
                idx_k = np.where(y_train == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
                # Balance
                # Case 1 (original case in Dir): Balance the number of sample per client
                proportions = np.array([p * (len(idx_j) < N / num_clients and j in clientidx_map[k]) for j, (p, idx_j) in enumerate(zip(proportions, idx_batch))])
                # Case 2: Don't balance
                #proportions = np.array([p * (j in label_netidx_map[k]) for j, (p, idx_j) in enumerate(zip(proportions, idx_batch))])
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                # process the remainder samples
                '''Note: Process the remainder data samples (yipeng, 2023-11-14).
                There are some cases that the samples of class k are not allocated completely, i.e., proportions[-1] < len(idx_k)
                In these cases, the remainder data samples are assigned to the last client in `clientidx_map[k]`.
                '''
                if proportions[-1] != len(idx_k):
                    for w in range(clientidx_map[k][-1], num_clients-1):
                        proportions[w] = len(idx_k)
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))] 
                min_size = min([len(idx_j) for idx_j in idx_batch])

        for j in range(num_clients):
            np.random.shuffle(idx_batch[j])
            dataidx_map[j] = idx_batch[j]
    
    else:
        raise NotImplementedError

    # assign data
    for client in range(num_clients):              # 对于每个客户端，根据字典分配的数据索引，获取其对应的图像数据和标签数据
        idxs = dataidx_map[client]
        X[client] = dataset_content[idxs]
        y[client] = dataset_label[idxs]

        for i in np.unique(y[client]):
            statistic[client].append((int(i), int(sum(y[client]==i))))
            # 计算每个客户端的样本数和每个类别的样本数

    del data        # 删除原始数据集以节省内存
    # gc.collect()

    for client in range(num_clients):     # 打印每个客户端的样本数和每个类别的样本数
        print(f"Client {client}\t Size of data: {len(X[client])}\t Labels: ", np.unique(y[client]))
        print(f"\t\t Samples of labels: ", [i for i in statistic[client]])
        print("-" * 50)

    return X, y, statistic


def split_data(X, y):        # 用于将每个客户端的数据集进一步划分为训练集和测试集
    # Split dataset
    train_data, test_data = [], []     # 初始化训练集和测试集列表
    num_samples = {'train':[], 'test':[]}   # 初始化训练集和测试集样本数列表

    for i in range(len(y)):            # 对于每个客户端，根据train_ratio划分训练集和测试集,并随机打乱
        X_train, X_test, y_train, y_test = train_test_split(
            X[i], y[i], train_size=train_ratio, shuffle=True)

        train_data.append({'x': X_train, 'y': y_train})    #将每个客户端的训练集和测试集保存到列表中
        num_samples['train'].append(len(y_train))            # 计算每个客户端的训练集样本数
        test_data.append({'x': X_test, 'y': y_test})
        num_samples['test'].append(len(y_test))

    print("Total number of samples:", sum(num_samples['train'] + num_samples['test']))
    print("The number of train samples:", num_samples['train'])
    print("The number of test samples:", num_samples['test'])
    print()
    # 打印总样本数和每个客户端的训练集和测试集样本数
    del X, y   # 删除原始数据集以节省内存
    # gc.collect()

    return train_data, test_data   # 返回训练集和测试集列表

def save_file(config_path, train_path, test_path, train_data, test_data, num_clients,       # 用于将生成的数据集和配置信息保存到文件中
                num_classes, statistic, niid=False, balance=True, partition=None):
    config = {                                        #创建一个字典，保存数据集的配置信息
        'num_clients': num_clients, 
        'num_classes': num_classes, 
        'non_iid': niid, 
        'balance': balance, 
        'partition': partition, 
        'Size of samples for labels in clients': statistic, 
        'alpha': alpha, 
        'batch_size': batch_size, 
    }

    # gc.collect()
    print("Saving to disk.\n")                # 打印提示信息，表示开始保存数据集到文件中

    for idx, train_dict in enumerate(train_data):       # 对于每个客户端的训练集，保存到文件中为.npz压缩文件格式
        with open(train_path + str(idx) + '.npz', 'wb') as f:
            np.savez_compressed(f, data=train_dict)
    for idx, test_dict in enumerate(test_data):
        with open(test_path + str(idx) + '.npz', 'wb') as f:
            np.savez_compressed(f, data=test_dict)
    with open(config_path, 'w') as f:                  # 将配置字典保存到文件中为.json格式
        ujson.dump(config, f)

    print("Finish generating dataset.\n")


class ImageDataset(Dataset):                        #  定义了一个名为ImageDataset的类，继承自torch.utils.data.Dataset，用于创建自定义图像数据集
    def __init__(self, dataframe, image_folder, transform=None):   # 初始化函数，接受一个包含文件名的dataframe，图像文件夹路径，以及可选的图像变换函数
        """
        Args:
            dataframe (pd.DataFrame): DataFrame containing file names
            image_folder (str): Path to the folder containing the images
            transform (callable, optional): Optional transform to be applied to the image
        """
        self.dataframe = dataframe
        self.image_folder = image_folder
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        # Get the file name from the DataFrame
        img_name = self.dataframe.iloc[idx]['file_name']
        img_label = self.dataframe.iloc[idx]['class']
        img_path = os.path.join(self.image_folder, img_name)
        
        # Load the image using PIL
        image = Image.open(img_path).convert('RGB')  # Ensure RGB if not grayscale
        
        if self.transform:
            image = self.transform(image)
        
        return image, img_label