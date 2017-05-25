# -*- coding: utf-8 -*-
import dlib
from skimage import io
import numpy as np

predictor_path = "/usr/local/lib/python2.7/dist-packages/dlib-19.4/python_examples/shape_predictor_68_face_landmarks.dat"
path="/home/kary/Projects/Face_Ex/ck/haarck/"

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

f=open("/home/kary/Projects/Face_Ex/ck/haarck/fileharrck.txt","r")
input_data=[]   #每张图片有136个x,y，所有图片的x,y都放到这个列表里面
output_data=[]
lines=f.readlines()
    #print line
for line in lines:
    if line:
        line=line.strip('\n')   #去除读取一行时末尾的换行符
        content=line.split(' ')
        tmp=content[0]
        srcimg=io.imread(path+tmp[35:58])
        dets=detector(srcimg,1)
        if dets==0:
            print "no dets"
            continue
        for k, d in enumerate(dets):
            shape = predictor(srcimg, d)
        for i in range(len(dets)):
            facepoint = np.array([[p.x, p.y] for p in predictor(srcimg, dets[i]).parts()])

        a = np.array(facepoint) #６８×２的数组 (x,y)
        b=content[1]    #注意b是str类型，要转化成int类型才能保存到np中
        #print b
        #print type(b)
        # 取第一列为x,第二列为y
        b_x = a[:, :1]
        b_y = a[:, 1:2]
        # 求x,y的标准差
        v_x = b_x.std()
        v_y = b_y.std()

        in_arr = []
        for num in b_x:
            x = (num - facepoint[30][0]) / v_x
            in_arr.append(x)
            # input_data.append(x)
            y = (num - facepoint[30][1]) / v_y
            in_arr.append(y)
            # input_data.append(y)
        input_data.append(in_arr)
        print  len(input_data) #每次以1为单位增长，input_data是列表
        #print type(output_data)
        output_data.append(int(b))
        #print type(output_data)

print "getdata ok!!!"
input_arr=np.array(input_data)
output_arr=np.array(output_data)
np.savetxt("/home/kary/Projects/Face_Ex/data1.txt",input_arr)
print "save input_txt ok!!!"
np.savetxt("/home/kary/Projects/Face_Ex/label.txt",output_arr)
print "save label_txt ok!!!"

f.close()





