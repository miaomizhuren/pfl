import numpy as np
import os
import sys
import random
import torch
import torchvision   # 计算机视觉库
import torchvision.transforms as transforms   # 数据预处理库
from utils.dataset_utils import check, separate_data, split_data, save_file  # 自定义函数


random.seed(1)# 设置随机种子
np.random.seed(1)# 设置numpy随机种子
num_clients = 20
dir_path = "MNIST/" # 作用是指定MNIST数据集的路径


# Allocate data to users
def generate_dataset(dir_path, num_clients, niid, balance, partition):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    # Setup directory for train/test data
    config_path = dir_path + "config.json" # 作用是指定MNIST数据集的配置文件路径
    train_path = dir_path + "train/"
    test_path = dir_path + "test/"

    if check(config_path, train_path, test_path, num_clients, niid, balance, partition):
        return   # 如果我发现你想要的数据集已经存在了，并且配置完全一样，那我就直接收工（return），不重新下载和切分数据了

    # # FIX HTTP Error 403: Forbidden
    # from six.moves import urllib
    # opener = urllib.request.build_opener()
    # opener.addheaders = [('User-agent', 'Mozilla/5.0')]
    # urllib.request.install_opener(opener)   爬虫爬数据伪装身份

    # Get MNIST data
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
  # 标准式起手  compose是多个transform的组合，第一个是ToTensor，将图片转化为tensor，第二个是Normalize，将图片归一化到[-1,1]
    trainset = torchvision.datasets.MNIST(
        root=dir_path+"rawdata", train=True, download=True, transform=transform)
    testset = torchvision.datasets.MNIST(
        root=dir_path+"rawdata", train=False, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=len(trainset.data), shuffle=False)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=len(testset.data), shuffle=False)
##数据加载，应用预处理转换，然后全都一口气提出来
    for _, train_data in enumerate(trainloader, 0):
        #其实一次就取完了，因为batch_size=len(trainset.data)=60000，写成循环为了符合迭代器语法规范
        trainset.data, trainset.targets = train_data
        #处理完了，把数据放回去
    for _, test_data in enumerate(testloader, 0):
        testset.data, testset.targets = test_data

    dataset_image = []
    dataset_label = []
    # 丝滑小连招
    dataset_image.extend(trainset.data.cpu().detach().numpy())
    # .cpu()把数据从显存（VRAM）搬运回内存（RAM） 其实一般都在CPU上，这边是为了保险
    # .detach()把tensor从计算图中分离出来，防止反向传播时计算梯度.PyTorch 的 Tensor 默认是会记录“身世”（Gradient 梯度）的，为了反向传播。但我们现在只是要原始数据值拿去分发，不需要它的求导历史。这就像“撕掉快递单上的历史记录，只留货物本身”。
    #  .numpy()把tensor转化为numpy数组。接下来的切分算法（separate_data）是基于 NumPy 写的，它不认识 PyTorch 的 Tensor。
    dataset_image.extend(testset.data.cpu().detach().numpy())
    #  .extend()是列表的合并操作，把两个列表合并成一个列表
    dataset_label.extend(trainset.targets.cpu().detach().numpy())
    dataset_label.extend(testset.targets.cpu().detach().numpy())
    #  下面两步把 Python 的 list（链表，内存不连续，访问慢）转换成 numpy.ndarray，为下一步的切分函数 separate_data 做好准备。那个函数需要进行复杂的矩阵索引操作
    dataset_image = np.array(dataset_image)
    dataset_label = np.array(dataset_label)

    num_classes = len(set(dataset_label))   #set()函数是去除重复元素，然后返回一个set对象，len()函数是返回集合的长度
    print(f'Number of classes: {num_classes}')
#  上面两步是自动获取数据集的类别数，并打印出来

    # # 以下代码是把数据集按类别分割成多个列表，但没用到
    # dataset = []
    # for i in range(num_classes):
    #     idx = dataset_label == i
    #     dataset.append(dataset_image[idx])

    X, y, statistic = separate_data((dataset_image, dataset_label), num_clients, num_classes, 
                                    niid, balance, partition, class_per_client=2)
    #   把那个合并后的 70,000 张图的大池子，按照规则切分给 20 个客户端
    #   class_per_client=2: 难度设定。如果开启了 Non-IID，这个参数规定了每个客户端只能看到 2 种数字
    #   x记录了每个client拥有的内容的下标，y记录了每个client拥有的数据的标签的下标
    #   statistic: 统计报表。它记录了每个客户端分到了什么数据。比如 Client 0: {数字1: 500张, 数字5: 400张}。这个报表通常会被打印出来或者存到 config 里，让你检查切分得是否符合预期。
    train_data, test_data = split_data(X, y)
    #   把切分好的 20 个客户端的训练集和测试集，分别存到 train_data 和 test_data 里
    save_file(config_path, train_path, test_path, train_data, test_data, num_clients, num_classes, 
        statistic, niid, balance, partition)
    #   把切分好的 20 个客户端的训练集和测试集，以及统计报表，存到文件里。


if __name__ == "__main__":
    niid = True if sys.argv[1] == "noniid" else False
    #检查你输入的第一个参数是不是单词 "noniid"
    balance = True if sys.argv[2] == "balance" else False
    #检查你输入的第二个参数是不是单词 "balance" balance=true就每人数据量一样
    partition = sys.argv[3] if sys.argv[3] != "-" else None
    #检查你输入的第三个参数是不是单词 "pat" "dir" "exdir" 或者是 "-"，如果是"-"，就不切分数据集了
    generate_dataset(dir_path, num_clients, niid, balance, partition)
    #调用generate_dataset函数，生成数据集