import sys
sys.path.append("../lib/")
import dataset
from dnn import *
import time
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage
np.random.seed(1)

try:
	train_file = sys.argv[1]
	test_file  = sys.argv[2]
except:
	sys.stderr.write("\tpython "+sys.argv[0]+" train_file test_file\n")
	sys.exit(-1)
# step 1 : 载入数据
train_x_orig, train_y, test_x_orig, test_y, classes = dataset.load_dataset(train_file, test_file)
# step 2 : 展示一个样本
#indx = 10
#plt.imshow(train_x_orig[indx])
#plt.show()
#print ("y = " + str(train_y[0,indx]) + ". It's a " + classes[train_y[0,indx]].decode("utf-8") +  " picture.")
# step 3 : 将图像展开成为向量
train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T
test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T
# step 4 : 归一化处理
train_x = train_x_flatten/255.
test_x = test_x_flatten/255.
print ("train_x's shape: " + str(train_x.shape))
print ("test_x's shape: " + str(test_x.shape))
# step 5 : 设置神经网络大小
n_x = 12288     # num_px * num_px * 3
n_h = 7
n_y = 1
layers_dims = (n_x, n_h, n_y)
# step 6 : 训练2层神经网络
#parameters = two_layer_model(train_x, train_y, layers_dims = (n_x, n_h, n_y), num_iterations = 2500, print_cost=False)
#train_predict = predict(train_x, parameters)
#print("train accuracy : %.2f%%" % (accuracy(train_predict, train_y)))
#test_predict = predict(test_x, parameters)
#print("test  accuracy : %.2f%%" % (accuracy(test_predict, test_y)))
# step 7 : 
layers_dims = [12288, 20, 7, 5, 1]
parameters = L_layer_model(train_x, train_y, layers_dims, num_iterations = 2500, print_cost = True)
train_predict = predict(train_x, parameters)
print("train accuracy : %.2f%%" % (accuracy(train_predict, train_y)))
test_predict = predict(test_x, parameters)
print("test  accuracy : %.2f%%" % (accuracy(test_predict, test_y)))
#print(parameters)
