
print("------------step 1 : 标量的sigmoid函数------------") 
import math
def basic_sigmoid(x):
	s = 1/(1+math.exp(-x))
	return s
print("sigmoid(3)(expected output) = 0.9526")
print("sigmoid(3)(real output)     = %.4f" % (basic_sigmoid(3)))

print("------------step 2 : 向量（矩阵）的sigmoid函数------------") 
import numpy as np
def sigmoid(x):
	s = 1/(1+np.exp(-x))
	return s
x = np.array([[1, 2, 3]])
y = sigmoid(x)
print("x.shape(the input  for sigmoid) = %s" % (str(x.shape)))
print("y.shape(the output for sigmoid) = %s" % (str(y.shape)))
print("sigmoid(%s) = %s" % (x, y))

print("------------step 3 : 向量（矩阵）的sigmoid函数求导------------") 
def sigmoid_derivative(x):
	s = 1/(1+np.exp(-x))
	ds = s * (1-s) # *是两个向量（矩阵）对应元素做乘法，非矩阵乘法
	return ds
x = np.array([1,2,3])
ds= sigmoid_derivative(x)
print("x.shape  = %s" % (x.shape))
print("ds.shape = %s" % (ds.shape))
print("sigmoid_derivative([1,2,3])(expected) = [ 0.19661193 0.10499359 0.04517666]")
print("sigmoid_derivative(%s)(real)     = %s" % (x, ds))

print("------------step 4 : 矩阵reshape------------")
def image2vector(image):
	nx,ny,nz = image.shape
	v  = image.reshape(nx*ny*nz,1)
	return v
def image2vector2(image):
	nx,ny,nz = image.shape
	v  = image.reshape(nx*ny, nz)
	return v
def image2vector3(image):
	nx,ny,nz = image.shape
	v  = image.reshape(nx, ny*nz)
	return v
def image2vector4(image):
	nx,ny,nz = image.shape
	v  = image.reshape(1, nx*ny*nz)
	return v
image = np.array([[[ 0.67826139,  0.29380381],
        [ 0.90714982,  0.52835647],
        [ 0.4215251 ,  0.45017551]],

       [[ 0.92814219,  0.96677647],
        [ 0.85304703,  0.52351845],
        [ 0.19981397,  0.27417313]],

       [[ 0.60659855,  0.00533165],
        [ 0.10820313,  0.49978937],
        [ 0.34144279,  0.94630077]]])
print("image.shape = "+str(image.shape))
print("image = "+str(image))
vec = image2vector(image)
print("image2vector(image) = " + str(vec))
print("image2vector2(image) = " + str(image2vector2(image)))
print("image2vector3(image) = " + str(image2vector3(image)))
print("image2vector4(image) = " + str(image2vector4(image)))
nx, ny, nz = image.shape
flag = True
for i in range(nx):
	for j in range(ny):
		for k in range(nz):
			if image[i,j,k] != vec[i*ny*nz+j*nz+k]:
				flag = False
				break
print("我对矩阵转向量的理解："+str(flag))

print("------------step 5 : 归一化------------")
def normalizeRows(x):
	x_norm = np.linalg.norm(x, ord=2, axis=1, keepdims=True)
	x = x/x_norm
	return x
x = np.array([
    [0, 3, 4],
    [1, 6, 4]])
print("x = "+str(x))
print("normalizeRows(x) = " + str(normalizeRows(x)))

print("------------step 5 : 广播（broadcasting）------------")
def softmax(x):
	x_exp = np.exp(x)
	x_sum = np.sum(x_exp, axis=1, keepdims=True)
	s     = x_exp/x_sum

	return s
x = np.array([
    [9, 2, 5, 0, 0],
    [7, 5, 0, 0 ,0]])
print("x = "+str(x))
print("softmax(x) = " + str(softmax(x)))
