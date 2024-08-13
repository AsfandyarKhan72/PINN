import os
os.environ["DDE_BACKEND"] = "tensorflow"
import tensorflow as tf

# Import necessary libraries
import numpy as np
import pandas as pd
import deepxde as dde  # Deep learning framework for solving differential equations
import matplotlib.pyplot as plt  # For creating static, animated, and interactive visualizations in Python

# Constants/Network Parameters
WIDTH = LENGTH = 1.0  
# Define geometry
geom = dde.geometry.Rectangle([0, 0], [WIDTH, LENGTH])  # 10x10 plate centered at (5, 5)

# Define constants for the PDEs
E = 200e9  # Young's modulus
G = 77   # Shear modulus
nu = 0.3 # Poisson's ratio

# def pde(X, Y):
def pde(x, y):
    u, v = y[:, 0:1], y[:, 1:2]
    
    u_xx = dde.grad.hessian(y, x, component=0, i=0, j=0)
    u_yy = dde.grad.hessian(y, x, component=0, i=1, j=1)
    u_xy = dde.grad.hessian(y, x, component=0, i=0, j=1)
    
    v_xx = dde.grad.hessian(y, x, component=1, i=0, j=0)
    v_yy = dde.grad.hessian(y, x, component=1, i=1, j=1)
    v_xy = dde.grad.hessian(y, x, component=1, i=0, j=1)

    eq1 = u_xx + (1-nu)/2 * u_yy + (1+nu)/2 * v_xy
    eq2 = v_yy + (1-nu)/2 * v_xx + (1+nu)/2 * u_xy
    
    return [eq1, eq2]

# Boundary Conditons
def boundary_right(X, on_boundary):
    x, _ = X
    return on_boundary and np.isclose(x, WIDTH)  # Check if on the right boundary

def boundary_left(X, on_boundary):
    x, _ = X
    return on_boundary and np.isclose(x, 0)  # Check if on the left boundary

def boundary_top(X, on_boundary):
    _, y = X
    return on_boundary and np.isclose(y, LENGTH)  # Check if on the upper boundary

def boundary_bottom(X, on_boundary):
    _, y = X
    return on_boundary and np.isclose(y, 0)  # Check if on the lower boundary


# Define Dirichlet and Neumann boundary conditions
def constraint_bottom(X):
    return np.zeros((len(X), 1))  # At the bottom, U and V are kept as zero

def constraint_top(X):
    return np.ones((len(X), 1)) * 0.001  # At the top, V is kept as 0.001

def func_zero(X):
    return np.zeros((len(X), 1))  # On the other boundaries, the derivative of U, V is kept at 0 (Neumann condition)

# Define boundary conditions for U
bc_U_l = dde.NeumannBC(geom, func=func_zero, on_boundary=boundary_left, component=0)  # Left boundary for U
bc_U_r = dde.NeumannBC(geom, func=func_zero, on_boundary=boundary_right, component=0)  # Right boundary for U
bc_U_up = dde.DirichletBC(geom, func=func_zero, on_boundary=boundary_top, component=0)  # Upper boundary for U
bc_U_low = dde.DirichletBC(geom, func=func_zero, on_boundary=boundary_bottom, component=0)  # Lower boundary for U

# Define boundary conditions for V
bc_V_l = dde.NeumannBC(geom, func=func_zero, on_boundary=boundary_left, component=1)  # Left boundary for V
bc_V_r = dde.NeumannBC(geom, func=func_zero, on_boundary=boundary_right, component=1)  # Right boundary for V
bc_V_up = dde.DirichletBC(geom, func=constraint_top, on_boundary=boundary_top, component=1)  # Upper boundary for V
bc_V_low = dde.DirichletBC(geom, func=func_zero, on_boundary=boundary_bottom, component=1)  # Lower boundary for V

# Define data for the PDEs
data = dde.data.PDE(geom, pde, [bc_U_l, bc_U_r, bc_U_up, bc_U_low, bc_V_l, bc_V_r, bc_V_up, bc_V_low], num_domain=10000, num_boundary=1000, num_test=1000)

# Define the neural network models for u and v
ARCHITECTURE = [2] + [128] * 4 + [2]
ACTIVATION = "tanh"  # Activation function
INITIALIZER = "Glorot uniform"  # Weights initializer

net = dde.maps.FNN(ARCHITECTURE, ACTIVATION, INITIALIZER)  # Feed-forward neural network

# Create the model for the PDE
model = dde.Model(data, net)


# Resetoring and training the model
#model.restore("/home/asfandyarkhan/PINNResearch/PINNResearch/trained_PINN_model-50005.ckpt")
#Compile the models with the chosen optimizer, learning rate, and loss weights
model.compile("adam", lr=1e-3, loss='MSE', loss_weights=[10, 10, 1, 1, 1, 1, 1, 1, 1, 1])
model.train(iterations=70000, batch_size=32)
dde.optimizers.config.set_LBFGS_options(maxcor=50, ftol=1e-6, gtol=1e-8, maxiter=5000, maxfun=50000, maxls=50)
model.compile("L-BFGS-B")
losshistory, train_state = model.train()

## Save the trained model
#model.save("./trained_PINN_model")

# Plot the loss history if needed
#dde.saveplot(losshistory, trainstate, issave=True, isplot=True)

# Set up the grid
nelx = 100  # Number of elements in x direction
nely = 100  # Number of elements in y direction
x = np.linspace(0, 1, nelx + 1)  # x coordinates
y = np.linspace(0, 1, nely + 1)  # y coordinates

# Prepare the data for the prediction
test_x, test_y = np.meshgrid(x, y)
test_domain = np.vstack((np.ravel(test_x), np.ravel(test_y))).T

# Predict Solution
predicted_solution = model.predict(test_domain)
predicted_solution_u = predicted_solution[:, 0].reshape(test_x.shape)
predicted_solution_v = predicted_solution[:, 1].reshape(test_x.shape)

# Plot the results
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Plot u
im1 = ax1.contourf(test_x, test_y, predicted_solution_u, levels=100, cmap='jet')
ax1.set_title('u (x-displacement)')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
fig.colorbar(im1, ax=ax1)

# Plot v
im2 = ax2.contourf(test_x, test_y, predicted_solution_v, levels=100, cmap='jet')
ax2.set_title('v (y-displacement)')
ax2.set_xlabel('x')
ax2.set_ylabel('y')
fig.colorbar(im2, ax=ax2)

plt.tight_layout()
plt.show()

# Convert to TensorFlow tensors
test_domain_tf = tf.convert_to_tensor(test_domain, dtype=tf.float32)

# Define a function to compute gradients
@tf.function
def compute_gradients(X):
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(X)
        Y = model.net(X)  # Use the neural network directly
        u, v = Y[:, 0], Y[:, 1]
    
    u_x = tape.gradient(u, X)[:, 0]
    u_y = tape.gradient(u, X)[:, 1]
    v_x = tape.gradient(v, X)[:, 0]
    v_y = tape.gradient(v, X)[:, 1]
    
    return u_x, u_y, v_x, v_y

# Compute gradients
u_x, u_y, v_x, v_y = compute_gradients(test_domain_tf)

# Convert back to numpy arrays
u_x = u_x.numpy().reshape(test_x.shape)
u_y = u_y.numpy().reshape(test_x.shape)
v_x = v_x.numpy().reshape(test_x.shape)
v_y = v_y.numpy().reshape(test_x.shape)

# Compute the terms
new_term1 = (E / (1 - nu**2) * (nu * u_x + v_y))
new_term2 = (E / (1 - nu**2) * (u_x + nu * v_y))


# Create a new figure with two subplots side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Plot new_term1
im1 = ax1.contourf(test_x, test_y, new_term1, levels=100, cmap='jet')
ax1.set_title('E/(1-nu^2) * (nu*u_x + v_y)')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
fig.colorbar(im1, ax=ax1, label='Stress (Pa)')

# Plot new_term2
im2 = ax2.contourf(test_x, test_y, new_term2, levels=100, cmap='jet')
ax2.set_title('E/(1-nu^2) * (u_x + nu*v_y)')
ax2.set_xlabel('x')
ax2.set_ylabel('y')
fig.colorbar(im2, ax=ax2, label='Stress (Pa)')

# Adjust layout and display the plot
plt.tight_layout()
plt.show()





