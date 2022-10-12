import pandas as pd
import numpy as np
import sys


class LogisticRegression:
    '''
    learning_rate:学习率
    iterations:迭代器
    w:权重矩阵
    mean : 平均值
    var : 方差
    '''
    def __init__(self, learning_rate, interations):
        self.learning_rate = learning_rate
        self.interations = interations
        self.w = []
        self.var = []
        self.mean = []
        
    #计算sigmoid函数
    def sigmoid(self, x):
        return 1.0 / (1 + np.exp(-x))
    
    #计算梯度
    def grad(self, w, x, y):
        return ((y - self.sigmoid(x @ w)).T @ x).T
    
    #进行批量归一化并训练模型
    def fit(self, train_x, train_y, loss):
        w = np.ones((train_x.shape[1] + 1, 1))
        for i in range(train_x.shape[1]):
            self.mean.append(np.mean(train_x.iloc[:, i]))
            self.var.append(np.var(train_x.iloc[:, i]))
            train_x.iloc[:, i] = (train_x.iloc[:, i] - self.mean[i]) / np.sqrt(self.var[i])
        
        train_x = np.c_[train_x, np.ones(train_x.shape[0])]

        loss_history = 0
        interation = 0
        
        for i in range(train_x.shape[0]):
            loss_history += -train_y[i] * (np.dot(w.T, train_x[i])) + np.log2(1 + np.exp(np.dot(w.T, train_x[i])))
        
        while interation < self.interations:
            interation += 100
            dl = self.grad(w, train_x, train_y)
            
            w = w + dl * self.learning_rate * 0.95
            loss_new = 0
            
            for i in range(train_x.shape[0]):
                loss_new += -train_y[i] * (np.dot(w.T, train_x[i])) + np.log2(1 + np.exp(np.dot(w.T, train_x[i])))
            loss.append(loss_new/train_x.shape[0])
            if abs(loss_history - loss_new) < 0.00001:
                print("***loss is ok***")
                break
            loss_history = loss_new
            print(str(interation) + "/" + str(self.interations))
            print("loss :" + str(loss_new/train_x.shape[0]))
            
        self.w = w
        
    def predict(self, test_x):
        for i in range(test_x.shape[1]):
            test_x.iloc[:, i] = (test_x.iloc[:, i] - self.mean[i]) / np.sqrt(self.var[i])
        
        test_x = np.c_[test_x, np.ones(test_x.shape[0])]
        
        p = self.sigmoid(test_x @ self.w).T
        p = p.flatten()
        #print(p.shape)
        credit = test_x[:,9]
        for i in range(p.shape[0]):
            if p[i] > 0.5:
                p[i] = 1
            else:
                p[i] = 0
        return p
    