import sys
import numpy as np
import h5py
import matplotlib.pyplot as plt

def load_dataset(train_file, test_file):
	train_dataset = h5py.File(train_file, "r")
	train_set_x_orig = np.array(train_dataset["train_set_x"][:])
	train_set_y_orig = np.array(train_dataset["train_set_y"][:])
	test_dataset = h5py.File(test_file, "r")
	test_set_x_orig = np.array(test_dataset["test_set_x"][:])
	test_set_y_orig = np.array(test_dataset["test_set_y"][:])
	classes = np.array(test_dataset["list_classes"][:])
	train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
	test_set_y_orig  = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

	return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes
def vectorize4images(X):
	return X.reshape(X.shape[0], -1).T
def normalize4images(X):
	return X/255
if __name__ == "__main__":
	try:
		train_file = sys.argv[1]
		test_file  = sys.argv[2]
	except:
		sys.stderr.write("\tpython "+sys.argv[0]+" train_file test_file\n")
		sys.exit(-1)
	# step 1 : 载入数据
	train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes = load_dataset(train_file, test_file)
	# step 2 : 展示数据规模
	print("训练样本(X.shape)={}".format(train_set_x_orig.shape))
	print("训练样本(Y.shape)={}".format(train_set_y_orig.shape))
	print("训练样本(Y[:,:10])={}".format(train_set_y_orig[:, :10]))
	print("测试样本(X.shape)={}".format(test_set_x_orig.shape))
	print("测试样本(Y.shape)={}".format(test_set_y_orig.shape))
	print("{}".format(classes))
	# step 3 : 展示一张图片
	indx = 25
	plt.imshow(train_set_x_orig[indx])
	print("y = " + str(train_set_y_orig[:, indx]) + ", it's a '" + classes[np.squeeze(train_set_y_orig[:, indx])].decode("utf8") + "' picture.")
	plt.show()
	# step 4 : 将图片三维矩阵向量化
	train_set_x_flatten = vectorize4images(train_set_x_orig)
	test_set_x_flatten  = vectorize4images(test_set_x_orig)
	print ("train_set_x_flatten shape: " + str(train_set_x_flatten.shape))
	print ("train_set_y shape: " + str(train_set_y_orig.shape))
	print ("test_set_x_flatten shape: " + str(test_set_x_flatten.shape))
	print ("test_set_y shape: " + str(test_set_y_orig.shape))
	print ("sanity check after reshaping: " + str(train_set_x_flatten[0:5,0]))
