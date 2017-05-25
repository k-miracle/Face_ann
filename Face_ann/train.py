# -*- coding: utf-8 -*-
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.externals import joblib
import os,sys

#f=open("/home/kary/PycharmProjects/age_det/model_score.txt",'a')
input_data=np.loadtxt("/home/kary/Projects/Face_Ex/data1.txt")
output_data=np.loadtxt("/home/kary/Projects/Face_Ex/label.txt")
print input_data.shape
print output_data.shape

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

# print len(X_train)
# print len(X_test)
# print len(y_train)
# print len(y_test)
#print input_data.shape #输出（11794,136）
    # mlp=MLPClassifier(hidden_layer_sizes=(100,600),max_iter=1000,alpha=1e-4,solver='sgd',
    #               verbose=10,tol=1e-4,random_state=1,learning_rate_init=.1,learning_rate='adaptive') #分类器

mlp = MLPClassifier(hidden_layer_sizes=(100,600), max_iter=1000, alpha=1e-4, solver='sgd',
                        verbose=10, tol=1e-8, random_state=1, learning_rate_init=.1, learning_rate='adaptive')  # 分类器

print "this is mlpclassifier"
#运行到this is mlpclassifier这一步

mlp.fit(X_train,y_train)
print "this is fit"
joblib.dump(mlp, "/home/kary/Projects/Face_Ex/train_model_1.pkl")  #保存训练好的模型到本地
print "完成模型的保存"

print("Test set score: %f" % mlp.score(X_test, y_test))