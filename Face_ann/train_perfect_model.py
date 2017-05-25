# -*- coding: utf-8 -*-
import numpy as np
from sklearn.neural_network import MLPClassifier
import os,sys

def Train():
    f=open("/home/gp/zk/ann/Face_Ex/model_score.txt",'a')
    input_data = np.loadtxt("/home/gp/zk/ann/Face_Ex/data1.txt")
    output_data = np.loadtxt("/home/gp/zk/ann/Face_Ex/label.txt")
    # print input_data.shape
    # print output_data.shape
    # X_train,X_test=input_data[:820],input_data[820:]
    # y_train,y_test=output_data[:820],output_data[820:]

    countall = 0

    # X_train,X_test=input_data[:820],input_data[820:]
    # y_train,y_test=output_data[:820],output_data[820:]
    X_train = []
    X_test = []
    y_train = []
    y_test = []
    for i in range(927):
        if countall % 10 == 0:
            X_test.append(input_data[countall])
            y_test.append(output_data[countall])
        else:
            X_train.append(input_data[countall])
            y_train.append(output_data[countall])
        countall += 1

    mlp = MLPClassifier(hidden_layer_sizes=(int(sys.argv[1]), int(sys.argv[2])), max_iter=1000, alpha=1e-4, solver='sgd',
                        verbose=10, tol=1e-9, random_state=1, learning_rate_init=.1, learning_rate='adaptive')  # 分类器

    print "this is mlpclassifier"
    #运行到this is mlpclassifier这一步

    mlp.fit(X_train,y_train)
    print "this is fit"
    # joblib.dump(mlp, "/home/kary/Projects/Age_Detect/train_model_1.pkl")  #保存训练好的模型到本地
    # print "完成模型的保存"

    # print("Training set score: %f" % mlp.score(X_train, y_train))
    # print("Test set score: %f" % mlp.score(X_test, y_test))
    f.write("hidden_layer_sizes: %d" %int(sys.argv[1]))
    f.write(" %d" %int(sys.argv[2]))
    f.write('\n')
    f.write("Test set score: %f" % mlp.score(X_test, y_test))
    f.write('\n')
    #print  'predict\t', mlp.predict(X_test)
    # 建立训练数据集，输入节点11794个
    # 建立神经网络，输入层11794,8，隐藏层：100,600
    #开始训练

if __name__=="__main__":
    Train()
