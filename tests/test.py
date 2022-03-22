import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CloughTocher2DInterpolator
from scipy import interpolate


def func(x, y):
    return x*2 + y


x_shift = 5
y_shift = -30

nx = 400
ny = 50
x = np.arange(0, nx, 1)
y = np.arange(0, ny, 1)
xv, yv = np.meshgrid(y, x)
values = func(xv, yv)

print(xv.shape, yv.shape, values.shape)
plt.pcolormesh(xv, yv, values, shading='auto')
plt.scatter(25, 200, c='#000000', s=10)
plt.colorbar()
plt.show()

points = np.stack([np.ravel(xv), np.ravel(yv)], axis=1)
point_values = np.ravel(values)
new_points = np.stack([xv.ravel() + x_shift, yv.ravel() + y_shift], axis=1)

for i in range(1):
    s = time.time()
    # interp = CloughTocher2DInterpolator(points, point_values)
    # interp = interpolate.interp2d(points[:, 1], points[:, 0], point_values, kind='cubic')
    interp = interpolate.RectBivariateSpline(x, y, values, kx=3, ky=3)
    # grid_z2 = interp(new_points)
    grid_z2 = interp(x + x_shift, y + y_shift)
    print(time.time() - s)

# print(xv.shape, yv.shape, grid_z2.reshape(ny, nx).shape)
plt.pcolormesh(xv + x_shift, yv + y_shift, grid_z2.reshape(nx, ny), shading='auto')
plt.scatter(25 + x_shift, 200 + y_shift, c='#000000', s=10)
plt.colorbar()
plt.show()
