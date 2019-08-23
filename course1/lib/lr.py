import sys
import numpy as np
import scipy
from scipy import ndimage
def sigmoid(z):
	s = 1/(1+np.exp(-z))
	return s
def initialize_with_zeros(dim):
	w = np.zeros((dim,1))
	b = 0
	assert(w.shape == (dim, 1))
	assert(isinstance(b, float) or isinstance(b,int))
	return w,b
def propagate(w, b, X, Y):
	m = X.shape[1]
	A = sigmoid(np.dot(w.T, X)+1)
	cost = -np.sum(Y*np.log(A)+(1-Y)*np.log(1-A))/m
	dz = A-Y
	dw = np.dot(X, dz.T)/m
	db = np.sum(dz)/m

	assert(dw.shape == w.shape)
	assert(db.dtype == float)
	cost = np.squeeze(cost)
	assert(cost.shape == ())
	grads = {"dw":dw,
			"db":db
	}
	return grads, cost
def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost=False):
	costs = []
	for i in range(num_iterations):
		grads, cost = propagate(w, b, X, Y)
		dw = grads["dw"]
		db = grads["db"]
		w  = w - learning_rate * dw
		b  = b - learning_rate * db
		if i%100 == 0:
			costs.append(cost)
		if print_cost and i%100 == 0:
			print("Cost after iteration %i: %f" % (i, cost))
	params = {"w":w,
			"b":b
	}
	grads  = {"dw":dw,
			"db":db
	}
	return params, grads, costs
def predict(w, b, X):
	m = X.shape[1]
	w = w.reshape(X.shape[0], 1)
	A = sigmoid(np.dot(w.T, X)+b)

	Y_prediction = np.array(A>0.5, dtype=np.float64)
#	Y_prediction = np.zeros((1, m))
#	for i in range(A.shape[1]):
#		if A[0, i] > 0.5:
#			Y_prediction[0, i] = 1
#		else:
#			Y_prediction[0, i] = 0
	assert(Y_prediction.shape == (1, m))

	return Y_prediction
def accuracy(Y_prediction, Y_label):
	return 100 - np.mean(np.abs(Y_prediction - Y_label))*100
def model(X_train, Y_train, X_test, Y_test, num_iterations=2000, learning_rate=0.5, print_cost=False):
	#print("X_train.shape"+str(X_train.shape))
	#print("Y_train.shape"+str(Y_train.shape))
	#print("X_test.shape"+str(X_test.shape))
	#print("Y_test.shape"+str(Y_test.shape))
	nx = X_train.shape[0]
	w, b = initialize_with_zeros(nx)
	parameters, grads, costs = optimize(w, b, X_train, Y_train, 
		num_iterations=num_iterations, 
		learning_rate=learning_rate, 
		print_cost=print_cost)
	
	w = parameters["w"]
	b = parameters["b"]
	print("b:"+str(b))
	print("w[0,0]:"+str(w[0,0]))
	Y_prediction_test  = predict(w, b, X_test)
	Y_prediction_train = predict(w, b, X_train)

	print("train accuracy: {}%".format(accuracy(Y_prediction_train, Y_train)))
	print("test_accuracy: {}%".format(accuracy(Y_prediction_test, Y_test)))

	d = {"costs":costs,
		"Y_prediction_test":Y_prediction_test,
		"Y_prediction_train":Y_prediction_train,
		"w":w,
		"b":b,
		"learning_rate":learning_rate,
		"num_iterations":num_iterations
	}
	return d
if __name__ == "__main__":
	# step 1 : sigmoid测试
	print("sigmoid([0,2]) = "+str(sigmoid(np.array([0,2]))))
	print("------------------------------------------------")
	# step 2 : 参数初始化测试
	dim = 2
	w,b = initialize_with_zeros(dim)
	print("w = "+str(w))
	print("b = "+str(b))
	print("------------------------------------------------")
	# step 3 : 单次迭代测试（前向传播-->代价计算-->反向传播）
	w, b, X, Y = np.array([[1.],[2.]]), 2., np.array([[1.,2.,-1.],[3.,4.,-3.2]]), np.array([[1,0,1]])
	grads, cost = propagate(w, b, X, Y)
	print ("dw = " + str(grads["dw"]))
	print ("db = " + str(grads["db"]))
	print ("cost = " + str(cost))
	print("------------------------------------------------")
	# step 4 : 完整训练过程
	params, grads, costs = optimize(w, b, X, Y, num_iterations= 100, learning_rate = 0.009, print_cost = False)
	print ("w = " + str(params["w"]))
	print ("b = " + str(params["b"]))
	print ("dw = " + str(grads["dw"]))
	print ("db = " + str(grads["db"]))
	print("------------------------------------------------")
	# step 5 : 预测
	w = np.array([[0.1124579],[0.23106775]])
	b = -0.3
	X = np.array([[1.,-1.1,-3.2],[1.2,2.,0.1]])
	print ("predictions = " + str(predict(w, b, X)))
	# step 6 : 完整测试
	import dataset
	import matplotlib.pyplot as plt
	train_file = "../data/train_catvnoncat.h5"
	test_file  = "../data/test_catvnoncat.h5"
	train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = dataset.load_dataset(train_file, test_file)
	train_set_x_flatten = dataset.vectorize4images(train_set_x_orig)
	test_set_x_flatten  = dataset.vectorize4images(test_set_x_orig)
	train_set_x = dataset.normalize4images(train_set_x_flatten)
	test_set_x  = dataset.normalize4images(test_set_x_flatten)
	d = model(train_set_x, train_set_y, 
		test_set_x, test_set_y, 
		num_iterations=4000, learning_rate=0.005,
		print_cost=True)
	#costs = np.squeeze(d["costs"])
	#print("costs.shape = "+str(costs.shape))
	#plt.plot(costs)
	#plt.ylabel("cost")
	#plt.xlabel("iterations(per handreds)")
	#plt.title("Learning rate = "+str(d["learning_rate"]))
	#plt.show()
	print("------------------------------------------------")

	# step 7 : 测试不同的学习率
	#learning_rates = [0.01, 0.005, 0.001, 0.0001]
	#models = {}
	#for i in learning_rates:
	#	print("learning rate is "+str(i))
	#	models[str(i)] = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations=1500, learning_rate=i, print_cost=False)
	#	print("------------------------------------------------")
	#for i in learning_rates:
	#	plt.plot(np.squeeze(models[str(i)]["costs"]), label=str(models[str(i)]["learning_rate"]))
	#plt.ylabel("cost")
	#plt.xlabel("iterations(hundreds)")
	#legend = plt.legend(loc="upper center", shadow=True)
	#frame = legend.get_frame()
	#frame.set_facecolor("0.9")
	#plt.show()

	# step 8 : 用自己的图片测试
	num_px = train_set_x_orig[0].shape[0]
	while True:
		print("your image file : ")
		infile = sys.stdin.readline().strip()
		image = np.array(ndimage.imread(infile, flatten=False))
		image = image/255
		print("image.shape"+str(image.shape))
		print("image.shape="+str(image.shape))
		my_image = scipy.misc.imresize(image, size=(num_px, num_px)).reshape((1, num_px*num_px*3)).T
		print("my_image.shape"+str(my_image.shape))
		my_predicted_image = predict(d["w"], d["b"], my_image)
		print("y = " + str(np.squeeze(my_predicted_image)) + ", your algorithm predicts a \"" + classes[int(np.squeeze(my_predicted_image)),].decode("utf-8") +  "\" picture.")
		plt.imshow(image)
