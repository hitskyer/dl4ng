import numpy as np
import h5py
import matplotlib.pyplot as plt

np.random.seed(1)
def initialize_parameters(n_x, n_h, n_y):
	np.random.seed(1)

	W1 = np.random.randn(n_h, n_x)*0.01
	b1 = np.zeros((n_h, 1))
	W2 = np.random.randn(n_y, n_h)*0.01
	b2 = np.zeros((n_y, 1))

	assert(W1.shape == (n_h, n_x))
	assert(b1.shape == (n_h, 1))
	assert(W2.shape == (n_y, n_h))
	assert(b2.shape == (n_y, 1))

	parameters = {
		"W1":W1,
		"b1":b1,
		"W2":W2,
		"b2":b2
	}
	return parameters
def initialize_parameters_deep(layer_dims):
	np.random.seed(3)
	parameters = {}
	L = len(layer_dims)
	for l in range(1, L):
		parameters["W"+str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1])*0.01
		parameters["b"+str(l)] = np.zeros((layer_dims[l], 1))
	for l in range(1, L):
		assert(parameters["W"+str(l)].shape == (layer_dims[l], layer_dims[l-1]))
		assert(parameters["b"+str(l)].shape == (layer_dims[l], 1))
	return parameters
def linear_forward(A, W, b):
	Z = np.dot(W, A) + b
	cache = (A, W, b)
	return Z, cache
def sigmoid(Z):
	A = 1/(1+np.exp(-Z))
	return A, Z
def relu(Z):
	A = (Z>0)*Z
	return A, Z
def linear_activation_forward(A_prev, W, b, activation):
	Z, linear_cache = linear_forward(A_prev, W, b)
	if activation == "sigmoid":
		A, activation_cache = sigmoid(Z)
	elif activation == "relu":
		A, activation_cache = relu(Z)
	else:
		sys.stderr.write("unknown activation function : %s\n" % (activation))
		sys.exit(-1)
	assert(A.shape == (W.shape[0], A_prev.shape[1]))
	cache = (linear_cache, activation_cache)

	return A, cache
def L_model_forward(X, parameters):
	caches = []
	A = X
	L = len(parameters)//2
	for l in range(1, L):
		A_prev = A
		A, cache = linear_activation_forward(A_prev, parameters["W"+str(l)], parameters["b"+str(l)], "relu")
		caches.append(cache)
	AL, cache = linear_activation_forward(A, parameters["W"+str(L)], parameters["b"+str(L)], "sigmoid")
	caches.append(cache)

	return AL, caches
def compute_cost(AL, Y):
	m = Y.shape[1]
	cost = -(np.dot(Y, np.log(AL).T)+
		np.dot(1-Y, np.log(1-AL).T))/m
	cost = np.squeeze(cost)
	assert(cost.shape == ())
	return cost
def linear_backward(dZ, cache):
	A_prev, W, b = cache
	m = A_prev.shape[1]
	dW = np.dot(dZ, A_prev.T)/m
	db = np.sum(dZ, axis=1, keepdims=True)/m
	dA_prev = np.dot(W.T, dZ)
	assert(dA_prev.shape == A_prev.shape)
	assert(dW.shape == W.shape)
	assert(db.shape == b.shape)
	return dA_prev, dW, db
def relu_backward(dA, Z):
	return dA * (Z>0)
def sigmoid_backward(dA, Z):
	A, tmp_cache = sigmoid(Z)
	return dA *A*(1-A)
def linear_activation_backward(dA, cache, activation):
	linear_cache, activation_cache = cache
	if activation == "relu":
		dZ = relu_backward(dA, activation_cache)
	elif activation == "sigmoid":
		dZ = sigmoid_backward(dA, activation_cache)
	else:
		sys.stderr.write("unknown activation function : %s\n" % (activation))
		sys.exit(-1)
	dA_prev, dW, db = linear_backward(dZ, linear_cache)
	return dA_prev, dW, db
def L_model_backward(AL, Y, caches):
	grads = {}
	L = len(caches)
	m = AL.shape[1]
	Y = Y.reshape(AL.shape)

	dAL = -(Y/AL - (1-Y)/(1-AL))
	current_cache = caches[L-1]
	grads["dA"+str(L-1)], grads["dW"+str(L)], grads["db"+str(L)] = linear_activation_backward(dAL, current_cache, "sigmoid")

	for l in reversed(range(L-1)):
		current_cache = caches[l]
		dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA"+str(l+1)], current_cache, "relu")
		grads["dA"+str(l)] = dA_prev_temp
		grads["dW"+str(l+1)] = dW_temp
		grads["db"+str(l+1)] = db_temp
	return grads
def update_parameters(parameters, grads, learning_rate):
	L = len(parameters)//2
	for l in range(L):
		parameters["W" + str(l+1)] = parameters["W" + str(l+1)]-learning_rate*grads["dW" + str(l+1)]
		parameters["b" + str(l+1)] = parameters["b" + str(l+1)]-learning_rate*grads["db" + str(l+1)]
	return parameters
def two_layer_model(X, Y, layers_dims, learning_rate=0.0075, num_iterations=3000, print_cost=False):
	np.random.seed(1)
	grads = {}
	costs = []
	m = X.shape[1]

	(n_x, n_h, n_y) = layers_dims
	parameters = initialize_parameters(n_x, n_h, n_y)
	W1 = parameters["W1"]
	b1 = parameters["b1"]
	W2 = parameters["W2"]
	b2 = parameters["b2"]

	for i in range(0, num_iterations):
		# step 1 : 前向传播
		A1, cache1 = linear_activation_forward(X, W1, b1, "relu")
		A2, cache2 = linear_activation_forward(A1, W2, b2, "sigmoid")
		# step 2 : 代价
		cost = compute_cost(A2, Y)
		# step 3 : 反向传播
		dA2 = -(Y/A2 - (1-Y)/(1-A2))
		dA1, dW2, db2 = linear_activation_backward(dA2, cache2, "sigmoid")
		dA0, dW1, db1 = linear_activation_backward(dA1, cache1, "relu")
		grads["dW1"] = dW1
		grads["db1"] = db1
		grads["dW2"] = dW2
		grads["db2"] = db2
		# step 4 : 参数更新
		parameters = update_parameters(parameters, grads, learning_rate)
		W1 = parameters["W1"]
		b1 = parameters["b1"]
		W2 = parameters["W2"]
		b2 = parameters["b2"]
		# step 5 : 打印代价
		if print_cost and i % 100 == 0:
			print("Cost after iteration {}: {}".format(i, np.squeeze(cost)))
		if print_cost and i % 100 == 0:
			costs.append(cost)
	if print_cost:
		plt.plot(np.squeeze(costs))
		plt.ylabel("cost")
		plt.xlabel('iterations (per hundreds)')
		plt.title("Learning rate =" + str(learning_rate))
		plt.show()
	return parameters
def predict(X, parameters):
	AL, _ = L_model_forward(X, parameters)
	return np.array(AL>0.5, dtype=np.float64)
def accuracy(Y_prediction, Y_label):
	return 100 - np.mean(np.abs(Y_prediction - Y_label))*100
def L_layer_model(X, Y, layers_dims, learning_rate=0.0075, num_iterations=3000, print_cost=False):
	np.random.seed(1)
	costs = []
	parameters = initialize_parameters_deep(layers_dims)
	for i in range(0, num_iterations):
		AL, caches = L_model_forward(X, parameters)
		cost = compute_cost(AL, Y)
		grads = L_model_backward(AL, Y, caches)
		parameters = update_parameters(parameters, grads, learning_rate)
		if print_cost and i % 100 == 0:
			print ("Cost after iteration %i: %f" %(i, cost))
		if print_cost and i % 100 == 0:
			costs.append(cost)
	if print_cost:
		plt.plot(np.squeeze(costs))
		plt.ylabel('cost')
		plt.xlabel('iterations (per hundreds)')
		plt.title("Learning rate =" + str(learning_rate))
		plt.show()
	
	return parameters
if __name__ == "__main__":
	# step 1 : 两层参数初始化
	parameters = initialize_parameters(3,2,1)
	print("W1 = " + str(parameters["W1"]))
	print("b1 = " + str(parameters["b1"]))
	print("W2 = " + str(parameters["W2"]))
	print("b2 = " + str(parameters["b2"]))
	print("---------------------------------------")
	# step 2 : L层参数初始化
	parameters = initialize_parameters_deep([5,4,3])
	print("W1 = " + str(parameters["W1"]))
	print("b1 = " + str(parameters["b1"]))
	print("W2 = " + str(parameters["W2"]))
	print("b2 = " + str(parameters["b2"]))
	print("---------------------------------------")
	# step 3 : 前向传播（线性加权部分）
	A = np.array([
		[1.62434536, -0.61175641],
		[-0.52817175, -1.07296862],
		[0.86540763, -2.3015387]])
	W = np.array([[1.74481176, -0.7612069, 0.3190391]])
	b = np.array([[-0.24937038]])
	Z, linear_cache = linear_forward(A, W, b)
	print("Z = " + str(Z))
	print("---------------------------------------")
	# step 4 : 前向传播（线性加权+激活函数）
	A_prev = np.array([
		[-0.41675785, -0.05626683], 
		[-2.1361961, 1.64027081],
		[-1.79343559, -0.84174737]])
	W = np.array([[0.50288142, -1.24528809, -1.05795222]])
	b = np.array([[-0.90900761]])
	A, linear_activation_cache = linear_activation_forward(A_prev, W, b, activation = "sigmoid")
	print("With sigmoid: A = " + str(A))
	A, linear_activation_cache = linear_activation_forward(A_prev, W, b, activation = "relu")
	print("With ReLU: A = " + str(A))
	print("---------------------------------------")
	# step 5 : L层前向传播
	X = np.array([
		[-0.31178367, 0.72900392, 0.21782079, -0.8990918],
		[-2.48678065, 0.91325152, 1.12706373, -1.51409323],
		[1.63929108, -0.4298936, 2.63128056, 0.60182225],
		[-0.33588161, 1.23773784, 0.11112817, 0.12915125],
		[0.07612761, -0.15512816, 0.63422534, 0.810655]])
	parameters = {
		'W1': np.array([[ 0.35480861,  1.81259031, -1.3564758 , -0.46363197,  0.82465384],
			[-1.17643148,  1.56448966,  0.71270509, -0.1810066 ,  0.53419953],
			[-0.58661296, -1.48185327,  0.85724762,  0.94309899,  0.11444143],
			[-0.02195668, -2.12714455, -0.83440747, -0.46550831,  0.23371059]]), 
		'b1': np.array([[ 1.38503523], [-0.51962709], [-0.78015214], [ 0.95560959]]), 
		'W2': np.array([[-0.12673638, -1.36861282,  1.21848065, -0.85750144],
			[-0.56147088, -1.0335199 ,  0.35877096,  1.07368134],
			[-0.37550472,  0.39636757, -0.47144628,  2.33660781]]), 
		'b2': np.array([[ 1.50278553],[-0.59545972],[ 0.52834106]]), 
		'W3': np.array([[ 0.9398248 ,  0.42628539, -0.75815703]]), 
		'b3': np.array([[-0.16236698]])
	}
	AL, caches = L_model_forward(X, parameters)
	print("AL = " + str(AL))
	print("Length of caches list = " + str(len(caches)))
	print("---------------------------------------")
	# step 6 : 代价函数
	Y = np.array([[1, 1, 1]])
	AL = np.array([[0.8, 0.9, 0.4]])
	print("cost = " + str(compute_cost(AL, Y)))
	print("---------------------------------------")
	# step 7 : 反向传播（线性加权部分）
	dZ = np.array([[1.62434536, -0.61175641]])
	linear_cache = (
		np.array([[-0.52817175, -1.07296862],
			[0.86540763, -2.3015387 ],
			[ 1.74481176, -0.7612069 ]]), 
		np.array([[0.3190391 , -0.24937038,  1.46210794]]), 
		np.array([[-2.06014071]])
	)
	dA_prev, dW, db = linear_backward(dZ, linear_cache)
	print ("dA_prev = "+ str(dA_prev))
	print ("dW = " + str(dW))
	print ("db = " + str(db))
	print("---------------------------------------")
	# step 8 : 反向传播（线性加权+激活）
	dAL = np.array([[-0.41675785, -0.05626683]])
	linear_activation_cache = ((
			np.array([[-2.1361961, 1.64027081],
				[-1.79343559, -0.84174737],
				[ 0.50288142, -1.24528809]]), 
			np.array([[-1.05795222, -0.90900761, 0.55145404]]), 
			np.array([[ 2.29220801]])
		), 
		np.array([[0.04153939, -1.11792545]])
	)
	dA_prev, dW, db = linear_activation_backward(dAL, linear_activation_cache, activation = "sigmoid")
	print ("sigmoid:")
	print ("dA_prev = "+ str(dA_prev))
	print ("dW = " + str(dW))
	print ("db = " + str(db) + "\n")

	dA_prev, dW, db = linear_activation_backward(dAL, linear_activation_cache, activation = "relu")
	print ("relu:")
	print ("dA_prev = "+ str(dA_prev))
	print ("dW = " + str(dW))
	print ("db = " + str(db))
	print("---------------------------------------")
	# step 9 : 反向传播（全部）
	AL = np.array([[1.78862847, 0.43650985]])
	Y_assess = np.array([[1, 0]])
	caches = (
		(
			(np.array([[ 0.09649747, -1.8634927 ],
				[-0.2773882 , -0.35475898],
				[-0.08274148, -0.62700068],
				[-0.04381817, -0.47721803]]), 
			np.array([[-1.31386475,  0.88462238,  0.88131804,  1.70957306],
				[0.05003364, -0.40467741, -0.54535995, -1.54647732],
				[0.98236743, -1.10106763, -1.18504653, -0.2056499 ]]), 
			np.array([[ 1.48614836], [0.23671627], [-1.02378514]])
			), 
			np.array([[-0.7129932, 0.62524497],
				[-0.16051336, -0.76883635],
				[-0.23003072,  0.74505627]])
		), 
		(
			(np.array([[ 1.97611078, -1.24412333],
				[-0.62641691, -0.80376609],
				[-2.41908317, -0.92379202]]), 
			np.array([[-1.02387576,  1.12397796, -0.13191423]]), 
			np.array([[-1.62328545]])
			), 
			np.array([[ 0.64667545, -0.35627076]])
		)
	)
	grads = L_model_backward(AL, Y_assess, caches)
	print("dW1 = "+str(grads["dW1"]))
	print("db1 = "+str(grads["db1"]))
	print("dA1 = "+str(grads["dA1"]))
	print("---------------------------------------")
	# step 10 : 更新参数
	parameters = {
		'W1': np.array([
			[-0.41675785, -0.05626683, -2.1361961 ,  1.64027081],
			[-1.79343559, -0.84174737,  0.50288142, -1.24528809],
			[-1.05795222, -0.90900761,  0.55145404,  2.29220801]]), 
		'b1': np.array([[ 0.04153939], [-1.11792545],[ 0.53905832]]), 
		'W2': np.array([[-0.5961597 , -0.0191305 ,  1.17500122]]), 
		'b2': np.array([[-0.74787095]])
	}
	grads = {
		'dW1': np.array([
			[ 1.78862847,  0.43650985,  0.09649747, -1.8634927 ],
			[-0.2773882 , -0.35475898, -0.08274148, -0.62700068],
			[-0.04381817, -0.47721803, -1.31386475,  0.88462238]]), 
		'db1': np.array([[ 0.88131804],[ 1.70957306],[ 0.05003364]]), 
		'dW2': np.array([[-0.40467741, -0.54535995, -1.54647732]]), 
		'db2': np.array([[ 0.98236743]])
	}
	parameters = update_parameters(parameters, grads, 0.1)

	print ("W1 = "+ str(parameters["W1"]))
	print ("b1 = "+ str(parameters["b1"]))
	print ("W2 = "+ str(parameters["W2"]))
	print ("b2 = "+ str(parameters["b2"]))
	print("---------------------------------------")
