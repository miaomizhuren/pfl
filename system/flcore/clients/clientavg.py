import copy
import torch
import numpy as np
import time
from flcore.clients.clientbase import Client


class clientAVG(Client):      # 继承自client类，意味着client类中包含的属性和方法都可以被clientAVG类使用
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)

    def train(self):    #  训练过程
        trainloader = self.load_train_data()    # 领原料，加载本地训练数据
        # self.model.to(self.device)32
        self.model.train()      # 开机器，开始训练
        
        start_time = time.time()      # 打卡，记录时间

        max_local_epochs = self.local_epochs
        if self.train_slow:     # 如果我是个慢客户端
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)   # 那我就训练慢点


# 深度学习三板斧：数据处理、模型设计、超参数调优
        for epoch in range(max_local_epochs):  #  这一轮我要训练多少个epoch
            for i, (x, y) in enumerate(trainloader): # # 遍历训练集
                if type(x) == type([]):    # 如果输入是列表，说明有多个输入
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)  # 转换设备
                y = y.to(self.device)      # 转换设备
                if self.train_slow:        # 如果我是个慢客户端
                    time.sleep(0.1 * np.abs(np.random.rand()))  # 模拟设备卡顿，强行睡一会儿
                output = self.model(x)     # 前向传播，算预测结果
                loss = self.loss(output, y)
                self.optimizer.zero_grad()# 梯度清零
                loss.backward()  #算梯度
                self.optimizer.step()  # 走一步（更新参数）

        # self.model.cpu()

        if self.learning_rate_decay:        # 如果我要学习率衰减，调整学习率，通常是越学越慢
            self.learning_rate_scheduler.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time   # 记录工时
