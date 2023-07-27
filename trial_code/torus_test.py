import numpy as np
import matplotlib.pyplot as plt


def plot_torus(precision, c, a):
    # U = np.linspace(0, 2*np.pi, precision)
    # V = np.linspace(0, 2*np.pi, precision)
    U = np.random.uniform(0, 2 * np.pi, size=(precision,))
    V = np.random.uniform(0, 2 * np.pi, size=(precision,))
    U, V = np.meshgrid(U, V)
    X = (c+a*np.cos(V))*np.cos(U)
    Y = (c+a*np.cos(V))*np.sin(U)
    Z = a*np.sin(V)
    return X, Y, Z


x, y, z = plot_torus(5, 2, 1)

fig = plt.figure()
ax = plt.axes(projection='3d')

# ax.plot_surface(x, y, z, antialiased=True, color='orange')
ax.scatter(x, y, z)
plt.show()