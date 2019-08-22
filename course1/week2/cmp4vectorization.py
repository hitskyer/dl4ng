import math
import time
import numpy as np

def cmp1():
	# 初始化
	a = np.random.rand(1000000)
	b = np.random.rand(1000000)
	# 向量版本
	tic = time.time()
	c   = np.dot(a,b)
	toc = time.time()
	print("a*b = %.4f" % (c))
	print("time for vectorization : %.4fms" % ((toc-tic)*1000))
	# 循环版本
	tic = time.time()
	c   = 0
	for i in range(1000000):
		c += a[i]*b[i]
	toc = time.time()
	print("a*b = %.4f" % (c))
	print("time for loop          : %.4fms" % ((toc-tic)*1000))
def cmp2():
	A = np.random.rand(1000, 1000)
	b = np.random.rand(1000)
	# 向量版本
	tic = time.time()
	c   = np.dot(A,b)
	toc = time.time()
	print("sum(A*b) = %.4f" % (np.sum(c)))
	print("time for vectorization : %.4fms" % ((toc-tic)*1000))
	# 循环版本
	tic = time.time()
	c   = np.zeros(1000)
	for i in range(1000):
		for j in range(1000):
			c[i] += A[i,j] * b[j]
	toc = time.time()
	print("sum(A*b) = %.4f" % (np.sum(c)))
	print("time for loop          : %.4fms" % ((toc-tic)*1000))
def cmp3():
	# 初始化
	a = np.random.rand(1000000)
	# 向量版本
	tic = time.time()
	c   = np.exp(a)
	toc = time.time()
	print("sum(exp(a)) = %.4f" % (np.sum(c)))
	print("time for vectorization : %.4fms" % ((toc-tic)*1000))
	# 循环版本
	tic = time.time()
	c   = 0
	for i in range(1000000):
		c += math.exp(a[i])
	toc = time.time()
	print("sum(exp(a)) = %.4f" % (c))
	print("time for loop          : %.4fms" % ((toc-tic)*1000))
cmp1()
print("------------------------------------")
cmp2()
print("------------------------------------")
cmp3()
