import copy    # 用于复制，通常用于复制模型以进行本地训练
import torch
import torch.nn as nn   # torch与torch.nn用于构建神经网络
import numpy as np
import os
from torch.utils.data import DataLoader   # 用于加载数据集
from sklearn.preprocessing import label_binarize  # 用于将标签转换为二进制编码
from sklearn import metrics   # 导入评价指标模块，比如AUC，准确率召回率啥的
from utils.data_utils import read_client_data   # 用于读取客户端数据


class Client(object):   # 客户端基本类
    """
    Base class for clients in federated learning.
    """

    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        torch.manual_seed(0)
        self.model = copy.deepcopy(args.model)   # 模型深拷贝，每个客户端都有自己的模型
        self.algorithm = args.algorithm            # 算法名称
        self.dataset = args.dataset               # 数据集名称
        self.device = args.device                # 设备名称
        self.id = id  # integer                             # 客户端编号
        self.save_folder_name = args.save_folder_name            # 保存模型的文件夹名称

        self.num_classes = args.num_classes                 # 类别数
        self.train_samples = train_samples                 # 训练样本数
        self.test_samples = test_samples                     # 测试样本数
        self.batch_size = args.batch_size
        self.learning_rate = args.local_learning_rate       # 本地学习率
        self.local_epochs = args.local_epochs                 # 本地训练轮数

        # check BatchNorm
        self.has_BatchNorm = False                       # 是否有BatchNormalization层
        for layer in self.model.children():              # 遍历模型的所有层
            if isinstance(layer, nn.BatchNorm2d):         # 如果包含BatchNormalization层，则设置has_BatchNorm为True
                self.has_BatchNorm = True
                break

        self.train_slow = kwargs['train_slow']            # 是否训练慢速模型
        self.send_slow = kwargs['send_slow']              # 是否发送慢速模型
        self.train_time_cost = {'num_rounds': 0, 'total_cost': 0.0}   # 训练时间统计
        self.send_time_cost = {'num_rounds': 0, 'total_cost': 0.0}    # 发送时间统计

        self.loss = nn.CrossEntropyLoss()      # 损失函数
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)# 优化器 并设置学习率
        self.learning_rate_scheduler = torch.optim.lr_scheduler.ExponentialLR(        # 学习率衰减
            optimizer=self.optimizer, 
            gamma=args.learning_rate_decay_gamma
        )
        self.learning_rate_decay = args.learning_rate_decay


    def load_train_data(self, batch_size=None):          #加载客户端训练数据
        if batch_size == None:
            batch_size = self.batch_size
        train_data = read_client_data(self.dataset, self.id, is_train=True)   # 使用read_client_data函数读取客户端数据
        return DataLoader(train_data, batch_size, drop_last=True, shuffle=True) # 返回一个DataLoader对象，设置drop_last=True以丢弃最后一个不完整的batch，shuffle=True表示打乱数据顺序

    def load_test_data(self, batch_size=None):           # 加载客户端测试数据
        if batch_size == None:
            batch_size = self.batch_size
        test_data = read_client_data(self.dataset, self.id, is_train=False)
        return DataLoader(test_data, batch_size, drop_last=False, shuffle=True)
        
    def set_parameters(self, model):                     #设置客户端模型参数
        for new_param, old_param in zip(model.parameters(), self.model.parameters()):   # 遍历传入模型和客户端模型的参数，将参数值复制到客户端模型中
            old_param.data = new_param.data.clone()

    def clone_model(self, model, target):             # 克隆模型
        for param, target_param in zip(model.parameters(), target.parameters()):  # 遍历模型参数，将参数值复制到目标模型中
            target_param.data = param.data.clone()
            # target_param.grad = param.grad.clone()

    def update_parameters(self, model, new_params):   # 更新客户端模型参数
        for param, new_param in zip(model.parameters(), new_params):  # 遍历客户端模型和传入模型的参数，将参数值更新到客户端模型中
            param.data = new_param.data.clone()

    def test_metrics(self):   # 测试模型性能
        testloaderfull = self.load_test_data()    # 加载测试数据集
        # self.model = self.load_model('model')
        # self.model.to(self.device)
        self.model.eval()    # 切换模型为测试模式

        test_acc = 0# 准确率
        test_num = 0# 样本数
        y_prob = []# 预测概率列表
        y_true = []# 真实标签列表
        
        with torch.no_grad():   # 在无梯度下进行测试
            for x, y in testloaderfull:    # 遍历测试数据集
                if type(x) == type([]):    # 如果输入数据为列表，则取第一个元素
                    x[0] = x[0].to(self.device)   # 将输入数据转移到设备
                else:
                    x = x.to(self.device)   # 将输入数据转移到设备
                y = y.to(self.device)     # 将标签转移到设备
                output = self.model(x)   # 前向传播

                test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()  # 计算当前批次的准确率并累加到test_acc中。
                test_num += y.shape[0]         # 累加当前批次的样本数量到test_num中。

                y_prob.append(output.detach().cpu().numpy()) # 将输出output转换为numpy数组，并添加到y_prob列表中。
                nc = self.num_classes
                if self.num_classes == 2:   # 根据类别数设置标签编码  如果类别数量为2，则nc增加1以处理二分类问题。
                    nc += 1
                lb = label_binarize(y.detach().cpu().numpy(), classes=np.arange(nc))
                if self.num_classes == 2:
                    lb = lb[:, :2]
                y_true.append(lb)

        # self.model.cpu()
        # self.save_model(self.model, 'model')

        y_prob = np.concatenate(y_prob, axis=0)
        y_true = np.concatenate(y_true, axis=0)         # 将所有批次的预测概率和真实标签合并为单个numpy数组。

        auc = metrics.roc_auc_score(y_true, y_prob, average='micro')  # 计算AUC
        
        return test_acc, test_num, auc   # 返回准确率，样本数，AUC

    def train_metrics(self):       # 训练模型性能
        trainloader = self.load_train_data()        # 加载训练数据集
        # self.model = self.load_model('model')
        # self.model.to(self.device)
        self.model.eval()         # 切换模型为测试模式

        train_num = 0            # 训练样本数
        losses = 0               # 损失值
        with torch.no_grad():       # 在无梯度下进行训练
            for x, y in trainloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)
                loss = self.loss(output, y)
                train_num += y.shape[0]
                losses += loss.item() * y.shape[0]

        # self.model.cpu()
        # self.save_model(self.model, 'model')

        return losses, train_num

    # def get_next_train_batch(self):
    #     try:
    #         # Samples a new batch for persionalizing
    #         (x, y) = next(self.iter_trainloader)
    #     except StopIteration:
    #         # restart the generator if the previous generator is exhausted.
    #         self.iter_trainloader = iter(self.trainloader)
    #         (x, y) = next(self.iter_trainloader)

    #     if type(x) == type([]):
    #         x = x[0]
    #     x = x.to(self.device)
    #     y = y.to(self.device)

    #     return x, y


    def save_item(self, item, item_name, item_path=None):     # 保存模型或其他数据
        if item_path == None:        # 如果没有指定保存路径，则使用默认路径
            item_path = self.save_folder_name    # 保存路径为save_folder_name
        if not os.path.exists(item_path):      # 如果保存路径不存在，则创建保存路径
            os.makedirs(item_path)         # 创建保存路径
        torch.save(item, os.path.join(item_path, "client_" + str(self.id) + "_" + item_name + ".pt"))  # 保存模型

    def load_item(self, item_name, item_path=None):     # 从文件加载指定模型或其他数据
        if item_path == None:        # 如果没有指定保存路径，则使用默认路径
            item_path = self.save_folder_name    # 保存路径为save_folder_name
        return torch.load(os.path.join(item_path, "client_" + str(self.id) + "_" + item_name + ".pt"))  # 加载模型

    # @staticmethod
    # def model_exists():
    #     return os.path.exists(os.path.join("models", "server" + ".pt"))
