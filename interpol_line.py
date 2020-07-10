import numpy as np
import matplotlib.pyplot as plt
size = int(input())
x =	np.random.uniform(0, size, size)
y = np.random.uniform(0, size, size)
z = np.random.uniform(0, size, size - 1)
print("x:", x, "y:", y, "z:", z)
#x = np.array([2, 5, 7])
#y = np.array([4, 25, 49])
#z = np.array([6])
n = len(z)

def ans(a, x, y, n):
	p = np.zeros(n)
	for i in range(n):
		p[i] = ((y[i + 1] - y[i]) / (x[i + 1] - x[i])) * (a[i] - x[i]) + y[i]
	return p

P = ans(z, x, y, n)
print(P)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(z, P)
ax.plot(x, y)
fig.savefig("linear_interpolation.png")
plt.show()