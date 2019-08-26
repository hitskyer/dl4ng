import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets
import sklearn.linear_model
import sys
sys.path.append("../lib/")
import dataset
def sigmoid(z):
	s = 1/(1+np.exp(-z))
	return s
def accuracy(Y_prediction, Y_label):
	return 100 - np.mean(np.abs(Y_prediction - Y_label))*100
def layer_sizes(X, Y):
	n_x = X.shape[0]
	n_h = 4
	n_y = Y.shape[0]
	return (n_x, n_h, n_y)
def initialize_parameters(n_x, n_h, n_y):
	np.random.seed(2)
	W1 = np.random.randn(n_h, n_x)*0.01
	b1 = np.zeros((n_h, 1))
	W2 = np.random.randn(n_y, n_h)*0.01
	b2 = np.zeros((n_y, 1))

	assert(W1.shape == (n_h, n_x))
	assert(b1.shape == (n_h, 1))
	assert(W2.shape == (n_y, n_h))
	assert(b2.shape == (n_y, 1))

	parameters = {"W1":W1,
		"b1":b1,
		"W2":W2,
		"b2":b2
	}
	return parameters
def forward_propagation(X, parameters):
	W1 = parameters["W1"]
	b1 = parameters["b1"]
	W2 = parameters["W2"]
	b2 = parameters["b2"]

	Z1 = np.dot(W1, X)+b1
	A1 = np.tanh(Z1)
	Z2 = np.dot(W2, A1)+b2
	A2 = sigmoid(Z2)

	assert(A2.shape == (1, X.shape[1]))

	cache = {
		"Z1":Z1,
		"A1":A1,
		"Z2":Z2,
		"A2":A2
	}
	return A2, cache
def compute_cost(A2, Y, parameters):
	m = Y.shape[1]
	
	cost = -(np.dot(Y, np.log(A2).T) + np.dot(1-Y, np.log(1-A2).T))/m
	cost = float(np.squeeze(cost))

	assert(isinstance(cost, float))

	return cost
def backward_propagation(parameters, cache, X, Y):
	m = X.shape[1]
	W1 = parameters["W1"]
	W2 = parameters["W2"]
	A1 = cache["A1"]
	A2 = cache["A2"]

	dZ2 = A2 - Y
	dW2 = np.dot(dZ2, A1.T)/m
	db2 = np.sum(dZ2, axis=1, keepdims=True)/m
	dZ1 = np.dot(W2.T, dZ2) * (1 - np.power(A1,2))
	dW1 = np.dot(dZ1, X.T)/m
	db1 = np.sum(dZ1, axis=1, keepdims=True)/m

	grads = {
		"dW1":dW1,
		"db1":db1,
		"dW2":dW2,
		"db2":db2
	}
	return grads
def update_parameters(parameters, grads, learning_rate=1.2):
	W1 = parameters["W1"]
	b1 = parameters["b1"]
	W2 = parameters["W2"]
	b2 = parameters["b2"]

	dW1 = grads["dW1"]
	db1 = grads["db1"]
	dW2 = grads["dW2"]
	db2 = grads["db2"]

	W1 = W1 - learning_rate * dW1
	b1 = b1 - learning_rate * db1
	W2 = W2 - learning_rate * dW2
	b2 = b2 - learning_rate * db2

	parameters = {
		"W1":W1,
		"b1":b1,
		"W2":W2,
		"b2":b2
	}
	return parameters
def nn_model(X, Y, n_h, num_iterations=10000, print_cost=False):
	np.random.seed(3)
	n_x = layer_sizes(X, Y)[0]
	n_y = layer_sizes(X, Y)[2]
	parameters = initialize_parameters(n_x, n_h, n_y)
	for i in range(0, num_iterations):
		A2, cache = forward_propagation(X, parameters)
		cost = compute_cost(A2, Y, parameters)
		grads = backward_propagation(parameters, cache, X, Y)
		parameters = update_parameters(parameters, grads)
		if print_cost and i%1000 == 0:
			print("Cost after iteration %i: %f" % (i, cost))
	return parameters
def predict(parameters, X):
	A2, cache = forward_propagation(X, parameters)
	predictions = np.array(A2>0.5, dtype=np.int64)
	return predictions
if __name__ == "__main__":
	# step 1 : 载入并展示数据
	X, Y = dataset.load_planar_dataset()
	m    = X.shape[1]
	#plt.scatter(X[0:1, :], X[1:, :], c=Y, s=40, cmap=plt.cm.Spectral)
	#plt.show()
	# step 2 : 训练LR模型并预测
	clf = sklearn.linear_model.LogisticRegressionCV()
	print(np.ravel(Y).shape)
	print(X.T.shape)
	clf.fit(X.T, np.ravel(Y))
	LR_predictions = clf.predict(X.T)
	print("Accuracy of logistic regression : %d" % (accuracy(LR_predictions, Y.T))+"%"+"(percentage of correctly labelled datapoints)")
	print("----------------------------------------------------")
	# step 3 : 初始化
	n_x = 2; n_h = 4; n_y = 1
	parameters = initialize_parameters(n_x, n_h, n_y)
	print("W1 = " + str(parameters["W1"]))
	print("b1 = " + str(parameters["b1"]))
	print("W2 = " + str(parameters["W2"]))
	print("b2 = " + str(parameters["b2"]))
	print("----------------------------------------------------")
	# step 4 : 前向传播
	X_assess = np.array([[1.62434536, -0.61175641, -0.52817175], [-1.07296862, 0.86540763, -2.3015387]])
	parameters = {
		'W1': np.array([[-0.00416758, -0.00056267],
			[-0.02136196,  0.01640271],
			[-0.01793436, -0.00841747],
			[ 0.00502881, -0.01245288]]), 
		'W2': np.array([[-0.01057952, -0.00909008,  0.00551454,  0.02292208]]), 
		'b1': np.array([[ 1.74481176],[-0.7612069 ],[ 0.3190391 ],[-0.24937038]]), 
		'b2': np.array([[-1.3]])
	}
	A2, cache = forward_propagation(X_assess, parameters)
	print(np.mean(cache['Z1']) ,np.mean(cache['A1']),np.mean(cache['Z2']),np.mean(cache['A2']))
	print("----------------------------------------------------")
	# step 5 : 计算代价函数
	A2 = np.array([[0.5002307, 0.49985831, 0.50023963]])
	Y_assess = np.array([[1, 0, 0]])
	parameters = {
		'W1': np.array([[-0.00416758, -0.00056267],
			[-0.02136196,  0.01640271],
			[-0.01793436, -0.00841747],
			[ 0.00502881, -0.01245288]]), 
		'W2': np.array([[-0.01057952, -0.00909008,  0.00551454,  0.02292208]]), 
		'b1': np.array([[ 0.], [ 0.], [ 0.], [ 0.]]), 
		'b2': np.array([[ 0.]])
	}
	print("cost = "+str(compute_cost(A2, Y_assess, parameters)))
	print("----------------------------------------------------")
	# step 6 : 反向传播
	parameters = {
		'W1': np.array([[-0.00416758, -0.00056267],
			[-0.02136196,  0.01640271],
			[-0.01793436, -0.00841747],
			[ 0.00502881, -0.01245288]]), 
		'W2': np.array([[-0.01057952, -0.00909008,  0.00551454,  0.02292208]]), 
		'b1': np.array([[ 0.], [ 0.], [ 0.], [ 0.]]), 
		'b2': np.array([[ 0.]])
	}
	cache = {
		'A1': np.array([[-0.00616578,  0.0020626 ,  0.00349619],
			[-0.05225116,  0.02725659, -0.02646251],
			[-0.02009721,  0.0036869 ,  0.02883756],
			[ 0.02152675, -0.01385234,  0.02599885]]), 
		'A2': np.array([[ 0.5002307 ,  0.49985831,  0.50023963]]), 
		'Z1': np.array([[-0.00616586,  0.0020626 ,  0.0034962 ],
			[-0.05229879,  0.02726335, -0.02646869],
			[-0.02009991,  0.00368692,  0.02884556],
			[ 0.02153007, -0.01385322,  0.02600471]]), 
		'Z2': np.array([[ 0.00092281, -0.00056678,  0.00095853]])
	}
	X_assess = np.array([[1.62434536, -0.61175641, -0.52817175], [-1.07296862,  0.86540763, -2.3015387]])
	Y_assess = np.array([[1, 0, 1]])
	grads = backward_propagation(parameters, cache, X_assess, Y_assess)
	print ("dW1 = "+ str(grads["dW1"]))
	print ("db1 = "+ str(grads["db1"]))
	print ("dW2 = "+ str(grads["dW2"]))
	print ("db2 = "+ str(grads["db2"]))
	print("----------------------------------------------------")
	# step 7 : 更新参数
	parameters = {
		'W1': np.array([[-0.00615039,  0.0169021 ],
			[-0.02311792,  0.03137121],
			[-0.0169217 , -0.01752545],
			[ 0.00935436, -0.05018221]]), 
		'W2': np.array([[-0.0104319 , -0.04019007,  0.01607211,  0.04440255]]), 
		'b1': np.array([[ -8.97523455e-07],[  8.15562092e-06],[  6.04810633e-07],[ -2.54560700e-06]]), 
		'b2': np.array([[  9.14954378e-05]])
	}
	grads = {
		'dW1': np.array([[ 0.00023322, -0.00205423],
			[ 0.00082222, -0.00700776],
			[-0.00031831,  0.0028636 ],
			[-0.00092857,  0.00809933]]), 
		'dW2': np.array([[ -1.75740039e-05,   3.70231337e-03,  -1.25683095e-03, -2.55715317e-03]]), 
		'db1': np.array([[  1.05570087e-07],[ -3.81814487e-06],[ -1.90155145e-07],[  5.46467802e-07]]), 
		'db2': np.array([[ -1.08923140e-05]])
	}
	parameters = update_parameters(parameters, grads)
	print("W1 = " + str(parameters["W1"]))
	print("b1 = " + str(parameters["b1"]))
	print("W2 = " + str(parameters["W2"]))
	print("b2 = " + str(parameters["b2"]))
	print("----------------------------------------------------")
	# step 8 : 完整的模型训练过程
	X_assess = np.array([[1.62434536, -0.61175641, -0.52817175], [-1.07296862, 0.86540763, -2.3015387]])
	Y_assess = np.array([[1, 0, 1]])
	parameters = nn_model(X_assess, Y_assess, 4, num_iterations=10000, print_cost=True)
	print("W1 = " + str(parameters["W1"]))
	print("b1 = " + str(parameters["b1"]))
	print("W2 = " + str(parameters["W2"]))
	print("b2 = " + str(parameters["b2"]))
	print("----------------------------------------------------")
	# step 9 : 预测
	parameters = {
		'W1': np.array([[-0.00615039,  0.0169021 ],
			[-0.02311792,  0.03137121],
			[-0.0169217 , -0.01752545],
			[ 0.00935436, -0.05018221]]), 
		'W2': np.array([[-0.0104319 , -0.04019007,  0.01607211,  0.04440255]]), 
		'b1': np.array([[ -8.97523455e-07], [  8.15562092e-06], [  6.04810633e-07],[ -2.54560700e-06]]), 
		'b2': np.array([[  9.14954378e-05]])
	}
	X_assess = np.array([[1.62434536, -0.61175641, -0.52817175], [-1.07296862,  0.86540763, -2.3015387 ]])
	predictions = predict(parameters, X_assess)
	print("predictions mean = " + str(np.mean(predictions)))
	print("----------------------------------------------------")
	# step 10 : 实测
	parameters = nn_model(X, Y, n_h = 4, num_iterations = 10000, print_cost=True)
	predictions = predict(parameters, X)
	print ('Accuracy: %d' % float((np.dot(Y,predictions.T) + np.dot(1-Y,1-predictions.T))/float(Y.size)*100) + '%')
	# step 10 : 调试隐藏层的神经单元数
	hidden_layer_sizes = [1, 2, 3, 4, 5, 20, 50]
	for i, n_h in enumerate(hidden_layer_sizes):
		parameters = nn_model(X, Y, n_h, num_iterations = 5000)
		predictions = predict(parameters, X)
		accuracy = float((np.dot(Y,predictions.T) + np.dot(1-Y,1-predictions.T))/float(Y.size)*100)
		print ("Accuracy for {} hidden units: {} %".format(n_h, accuracy))
		print("----------------------------------------------------")
