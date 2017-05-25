# -*- coding: utf-8 -*-
import numpy as np
from sklearn.externals import joblib
from sklearn.manifold import TSNE
import pandas as pd
import matplotlib.pyplot as plt
from time import time

# def sigmoid(z):
#     """The sigmoid function."""
#     return 1.0/(1.0+np.exp(-z))
def relu(z):
    return np.maximum(z, 0.0)   #注意relu对np函数的特殊定义法

input_data=np.loadtxt("/home/kary/Projects/Face_Ex/data1.txt")
output_data=np.loadtxt("/home/kary/Projects/Face_Ex/label.txt")
# print input_data.shape
# print output_data.shape
# X_train,X_test=input_data[:820],input_data[820:]
# y_train,y_test=output_data[:820],output_data[820:]

count_train = 0
count_validation = 0
countall = 0

# X_train,X_test=input_data[:820],input_data[820:]
# y_train,y_test=output_data[:820],output_data[820:]
X_train=[]
X_test=[]
y_train=[]
y_test=[]
for i in range(927):
    if countall % 10 == 0:
        X_test.append(input_data[countall])
        y_test.append(output_data[countall])
    else:
        X_train.append(input_data[countall])
        y_train.append(output_data[countall])
    countall += 1

mlp = joblib.load("/home/kary/Projects/Face_Ex/train_model_1.pkl") #加载训练好保存在本地的模型

out_data=[]
for x in X_test:
    activation=x    #输入值
    activations=[activation]
    zs=[]   #输出值
    z1=np.dot(activation,mlp.coefs_[0])+mlp.intercepts_[0]
    zs.append(z1)
    activation = relu(z1)
    activations.append(activation)
    # print z1.shape    #1*100
    # print type(z1)

    z2=np.dot(activation,mlp.coefs_[1])+mlp.intercepts_[1]
    zs.append(z2)
    activation =relu(z2)
    activations.append(activation)
    # print z2.shape    ＃1*600
    # print type(z2)    #array

    out_data.append(z2)
    print len(out_data)

# out_arr=np.array(out_data)
# print out_arr.shape #93*600

#tsne特征可视化
#第一种
out_label=mlp.predict(X_test)   #得到预测的标签
print type(out_label)
tsne=TSNE()
X_tsne = tsne.fit_transform(out_data)  #对特征，进行数据降维
print("finishe!")
plt.figure(figsize=(12, 6))
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=out_label)  #X_tsne已经降到二维了，0为x坐标，1为y坐标，
plt.colorbar()
plt.show()

# #第二种
# X_label=mlp.predict(X_test)
# print type(X_label)
# print X_label
# tsne=TSNE()
# X_tsne = tsne.fit_transform(X_test)  #进行数据降维
# X_label=mlp.predict(X_test)
# for point,lab in zip(X_tsne,X_label):
#     print point,lab
#     if(lab==0):
#         plt.plot(point[0],point[1],'yx')    #0-2,yx
#     elif(lab==1):
#         plt.plot(point[0],point[1],'go')    #4-6,go
#     elif(lab == 2):
#         plt.plot(point[0],point[1],'r*')    #8-13,r*
#     elif(lab == 3):
#         plt.plot(point[0],point[1],'cx')    #15-20,cx
#     elif(lab == 4):
#         plt.plot(point[0],point[1],'k*')    #25-32,k*
#     elif(lab == 5):
#         plt.plot(point[0],point[1],'bx')    #38-43,bx
#     elif(lab == 6):
#         plt.plot(point[0],point[1],'r+')    #48-53,r+　较少
# plt.show()
# print "finish"


#mlp.predict(X_test)
# print("Test set score: %f" % mlp.score(X_test, y_test))
# tt=mlp.predict(X_test)

# print type(tt)
# print tt.shape  #输出(3794,)：因为取的是8000～11794的数据集，即３７９４个元素对应３７９４个标签
# x=tt.reshape(1,3794)
# print x.shape
# print  'predict\t', mlp.predict(X_test)
# print tt
# out_data=[]
# for x in X_test:
#     activation=x    #输入值
#     activations=[activation]
#     zs=[]   #输出值
#     z1=np.dot(activation,mlp.coefs_[0])+mlp.intercepts_[0]
#     zs.append(z1)
#     activation = relu(z1)
#     activations.append(activation)
#     # print z1.shape    #1*180
#     # print type(z1)
#
#     z2=np.dot(activation,mlp.coefs_[1])+mlp.intercepts_[1]
#     zs.append(z2)
#     activation =relu(z2)
#     activations.append(activation)
#     # print z2.shape    ＃1*700
#     # print type(z2)    #array
#
#     out_data.append(z2)
#     print len(out_data)
#
# out_arr=np.array(out_data)
# print out_arr.shape #3794*700