import os
os.environ["DDE_BACKEND"] = "tensorflow"

import deepxde as dde
import numpy as np
from deepxde.backend import tf
import matplotlib.pyplot as plt

# Define the elasticity PDEs
def pde(x, u):
    nu = 0.3  # Poisson's ratio

    U = u[:, 0:1]
    V = u[:, 1:2]

    U_xx = dde.grad.hessian(u, x, component=0, i=0, j=0)
    U_yy = dde.grad.hessian(u, x, component=0, i=1, j=1)
    V_xx = dde.grad.hessian(u, x, component=1, i=0, j=0)
    V_yy = dde.grad.hessian(u, x, component=1, i=1, j=1)
    U_xy = dde.grad.hessian(u, x, component=0, i=0, j=1)
    V_xy = dde.grad.hessian(u, x, component=1, i=0, j=1)

    eq1 = U_xx + (1 - nu) / 2 * U_yy + (1 + nu) / 2 * V_xy
    eq2 = V_yy + (1 - nu) / 2 * V_xx + (1 + nu) / 2 * U_xy

    return [eq1, eq2]

# Geometry
geom = dde.geometry.Rectangle([0, 0], [1, 1])

# Boundary conditions
def bottom_boundary(x, on_boundary):
    return on_boundary and np.isclose(x[1], 0)

def top_boundary(x, on_boundary):
    return on_boundary and np.isclose(x[1], 1)

def left_boundary(x, on_boundary):
    return on_boundary and np.isclose(x[0], 0)

def right_boundary(x, on_boundary):
    return on_boundary and np.isclose(x[0], 1)

# Dirichlet BC at the bottom
bc1 = dde.DirichletBC(geom, lambda x: 0, bottom_boundary, component=0)  # U = 0 at bottom
bc2 = dde.DirichletBC(geom, lambda x: 0, bottom_boundary, component=1)  # V = 0 at bottom

# Dirichlet BC at the top
bc3 = dde.DirichletBC(geom, lambda x: 0.001, top_boundary, component=1)  # V = 0.001 at top

# Neumann BC on left and right boundaries (derivative is zero)
bc4 = dde.NeumannBC(geom, lambda x: 0, left_boundary, component=0)  # dU/dx = 0 on left
bc5 = dde.NeumannBC(geom, lambda x: 0, right_boundary, component=0)  # dU/dx = 0 on right
bc6 = dde.NeumannBC(geom, lambda x: 0, left_boundary, component=1)  # dV/dx = 0 on left
bc7 = dde.NeumannBC(geom, lambda x: 0, right_boundary, component=1)  # dV/dx = 0 on right


# Combine boundary conditions and initial conditions
data = dde.data.PDE(
    geom,
    pde,
    [bc1, bc2, bc3, bc4, bc5, bc6, bc7],
    num_domain=10000,
    num_boundary=500,
)

# Define neural network
net = dde.maps.FNN([2] + [60] * 5 + [2], "tanh", "Glorot uniform")

# Define model
model = dde.Model(data, net)

# Compile and train the model
model.compile("adam", lr=1e-3)
losshistory, train_state = model.train(iterations=50000, batch_size=32)
model.compile("L-BFGS")
losshistory, train_state = model.train()

# Plot the results
dde.saveplot(losshistory, train_state, issave=True, isplot=True)



# Evaluate the model on a grid
x = np.linspace(0, 1, 200)
y = np.linspace(0, 1, 200)
X, Y = np.meshgrid(x, y)
X_flat = X.flatten()[:, None]
Y_flat = Y.flatten()[:, None]
XY = np.hstack((X_flat, Y_flat))

U_V_pred = model.predict(XY)
U_pred = U_V_pred[:, 0].reshape(X.shape)
V_pred = U_V_pred[:, 1].reshape(X.shape)

# Plot the contour plot for U
plt.figure(figsize=(8, 6))
plt.contourf(X, Y, U_pred, levels=100, cmap="jet")
plt.colorbar(label="Displacement in X direction (U)")
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Contour plot of U (Displacement in X direction)")
plt.savefig("contour_U.png")
plt.show()

# Plot the contour plot for V
plt.figure(figsize=(8, 6))
plt.contourf(X, Y, V_pred, levels=100, cmap="jet")
plt.colorbar(label="Displacement in Y direction (V)")
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Contour plot of V (Displacement in Y direction)")
plt.savefig("contour_V.png")
plt.show()