import numpy as np
from scipy.interpolate import lagrange
from numpy.polynomial.polynomial import Polynomial
import matplotlib as plt
size = int(input())
x =	np.random.uniform(0, size, size)
y = np.random.uniform(0, size, size)
z = np.random.uniform(0, size, size - 1)
print("x:", x, "y:", y, "z:", z)
#x = np.array([0, 1, 2, 3])
#y = np.array([-2, -5, 0, -4])
#z = np.array([4])
n = len(x)


poly = lagrange(x, y)
dif.append(poly(z[i]))


def phi(i, z):
	p = 1
	for j in range(0, n):
		if i != j:
			p *= (z[0] - x[j]) / (x[i] - x[j])
	return p
	
def P(z):
	s = 0
	for i in range(n):
		s += y[i] * phi(i, z)
	return s			

print(P(z))	
plt.show()