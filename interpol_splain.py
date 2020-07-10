import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
size = int(input())
x =	np.random.uniform(0, size, size)
y = np.random.uniform(0, size, size)
z = np.random.uniform(0, size, size - 1)
print("x:", x)
print("y:", y)
print("z:", z)
#x = np.array([2, 5])
#y = np.array([4, 25])
#z = np.array([3])
#x = np.array([1, 2, 3, 4, 5, 6])
#y = np.array([1, 4, 9, 16, 25, 36])
#z = np.array([7, 8])
n = len(z)

def sweep(n,a, b, c, f):	
	alpha = [0] * (n + 1)
	beta = [0] * (n + 1)
	x = [0] * n
	a[0] = 0
	c[n - 1] = 0
	alpha[0] = 0
	beta[0] = 0
	for i in range(0, n):
		d = a[i] * alpha[i] + b[i]
		alpha[i + 1] = -c[i] / d
		beta[i + 1] = (f[i] - a[i] * beta[i]) / d
	x[n - 1] = beta[n]
	for i in range(n - 2, -1, -1):
		x[i] = alpha[i + 1] * x[i + 1] + beta[i + 1]
	return(x)

def generateSpline(x, y):
	n = x.shape[0] - 1
	A = np.zeros(n + 1)
	B = np.zeros(n + 1)
	C = np.zeros(n + 1)
	D = np.zeros(n + 1)
	h = (x[n] - x[0]) / n
	a = np.array([0] + [1] * (n - 1) + [0])
	b = np.array([1] + [4] * (n - 1) + [1])
	c = np.array([0] + [1] * (n - 1) + [0])
	f = np.zeros(n + 1)
	for i in range(1, n):
		f[i] = 3 * (y[i - 1] - 2 * y[i] + y[i + 1]) / (h ** 2)
	s = sweep(n + 1, a, b, c, f)
	
	for i in range(0, n):
		B[i] = s[i]
		A[i] = (B[i + 1] - B[i]) / (3 * h)
		C[i] = (y[i + 1] - y[i]) / h - (B[i + 1] + 2 * B[i]) * h / 3
		D[i] = y[i]
	return A, B, C, D		

def ans(a, A, B, C, D, x, n):
	p = np.zeros(n)
	for i in range(n):
		p[i] = A[i] * ((a[i] - x[i]) ** 3) + B[i] * ((a[i] - x[i]) ** 2) + C[i] * (a[i] - x[i]) + D[i]
	return p

A, B, C, D = generateSpline(x, y)
P = ans(z, A, B, C, D, x, n)
print(P)

plt.plot(x,y)
plt.show()
