import os
# Set backend
os.environ["DDE_BACKEND"] = "tensorflow"
import tensorflow as tf
import deepxde as dde
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from scipy.interpolate import RegularGridInterpolator
from deepxde.callbacks import Callback
import pandas as pd

BATCH_SIZE = 32  # Batch size
#LEARNING_RATE = 1e-3  # Learning rate

ITERATIONS_A = 20000  # Number of training iterations
ITERATIONS_LBFGS = 20000  # Number of training iterations
ITERATIONS_A2 = 20000  # Number of training iterations
ITERATIONS_LBFGS2 = 10000  # Number of training iterations 

T_Start = 0
TIME_STEP = 0.25
T_End = 0.25


# Define geometry and time domains
WIDTH = LENGTH = 1  # Domain size
geom = dde.geometry.Rectangle([0, 0], [WIDTH, LENGTH])  # Geometry domain
time_domain = dde.geometry.TimeDomain(T_Start, T_End)
geomtime = dde.geometry.GeometryXTime(geom, time_domain)



# Parameters
c1 = 0.0001  # Adjusted based on the paper
c2_trainable = tf.Variable(10.0, dtype=tf.float32) # Heat equation 1D

def allen_cahn_2D (x, y):  # Allen Cahn 2D Mattey with single output 
    h = y
    h_t = dde.grad.jacobian(h, x, i=0, j=2)
    # Second derivatives of phi with respect to x and y
    phi_xx = dde.grad.hessian(h, x, i=0, j=0)
    phi_yy = dde.grad.hessian(h, x, i=1, j=1)
    phi_terms = phi_xx + phi_yy
    # Calculating the modified term for the Laplacian
    modified_phi_term = (c1 * phi_terms)- c2_trainable * (h ** 3 - h)
    eq1 = h_t - modified_phi_term
    return eq1

# Initial condition Allen Cahn
def init_condition_AC_2D(X):
    x1, x2 = X[:, 0:1], X[:, 1:2]
    return np.sin(4 * np.pi * x1) * np.cos(4 * np.pi * x2)

# Boundary Conditions
def on_boundary_x(X, on_boundary):
    x, _, _ = X
    return on_boundary and np.isclose(x, 0) or np.isclose(x, WIDTH)  # Check if on the left boundary
def on_boundary_y(X, on_boundary):
    _, y, _ = X
    return on_boundary and np.isclose(y, 0) or np.isclose(y, LENGTH)  # Check if on the left boundary

# Periodic Boundary Conditions for 'h' in x and y directions
bc_h_x = dde.icbc.PeriodicBC(geomtime, component=0, derivative_order=0, component_x=0, on_boundary=on_boundary_x)
bc_h_y = dde.icbc.PeriodicBC(geomtime, component=0, derivative_order=0, component_x=1, on_boundary=on_boundary_y)

# Periodic Boundary Conditions for the derivative of 'h' in x and y directions
bc_h_deriv_x = dde.icbc.PeriodicBC(geomtime, component=0, derivative_order=1, component_x=0, on_boundary=on_boundary_x)
bc_h_deriv_y = dde.icbc.PeriodicBC(geomtime, component=0, derivative_order=1, component_x=1, on_boundary=on_boundary_y)

# Initial Condition 
initial_condition_AC_2d = dde.icbc.IC(geomtime, init_condition_AC_2D, lambda _, on_initial: on_initial)


# Load the data from the text file
data = np.loadtxt('/home/asfandyarkhan/PINN/data/data_ac_2d.txt')

# Split the data into spatial coordinates (x, y), time (t), and solution values (h)
observed_x = data[:, 0:1]  # The x column (as 2D array)
observed_y = data[:, 1:2]  # The y column (as 2D array)
observed_t = data[:, 2:3]  # The t column (as 2D array)
observed_h = data[:, 3:]   # The h (solution) column (as 2D array)

# Combine x, y, and t to create the observation points in the space-time domain
observed_xyt = np.hstack((observed_x, observed_y, observed_t))

# Define the PointSetBC using the observed points and solution values
observe_h_AC_2D = dde.icbc.PointSetBC(observed_xyt, observed_h, component=0)

# Data for AC 2D
data_AC_2D_inverse = dde.data.TimePDE(
    geomtime,
    allen_cahn_2D,
    [bc_h_x, bc_h_y, bc_h_deriv_x, bc_h_deriv_y, initial_condition_AC_2d, observe_h_AC_2D],  # Include observe_h here
    num_domain=30000,
    num_boundary=1600,
    num_initial=4096,
    anchors=observed_xyt,  # Make sure observed_xt is used as anchors if necessary
    num_test=50000,
)

# Your file path
file_path = "/home/asfandyarkhan/PINN/data/losses_simple.txt"
# Check if file exists and delete it
if os.path.exists(file_path):
    os.remove(file_path)
    print(f"Removed existing file: {file_path}")

class SimpleLossTrackingCallback(Callback):
    def __init__(self, every_n_epochs=1000, file_path="/home/asfandyarkhan/PINN/data/losses_simple.txt"):
        super(SimpleLossTrackingCallback, self).__init__()
        self.every_n_epochs = every_n_epochs
        self.file_path = file_path
        
        # Ensure the directory exists
        os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
        
        # Check if file exists and is not empty, if so, skip writing the header
        if not os.path.exists(self.file_path) or os.stat(self.file_path).st_size == 0:
            with open(self.file_path, "w") as f:
                f.write("Epoch,PDE Loss,bc_h_x Loss,bc_h_y Loss,bc_h_deriv_x Loss,bc_h_deriv_y Loss,IC Loss,Observe Loss\n")

    def on_epoch_end(self):
        if self.model.train_state.step % self.every_n_epochs == 0 or self.model.train_state.step == 1:
            current_losses = self.model.train_state.loss_train
            loss_str = ",".join(map(str, current_losses))
            with open(self.file_path, "a") as f:
                f.write(f"{self.model.train_state.step},{loss_str}\n")

iterations_list = [0]  # Starting with iteration 0
gamma_values = [c2_trainable.value().numpy()]  # Assuming this is how you access the value of your variable

# Network Architecure
net = dde.nn.FNN([3] + [128] * 6 + [1], "tanh", "Glorot normal")
#net = dde.nn.FNN([3] + [60] * 6 + [1], "tanh", "Glorot normal")
variable = dde.callbacks.VariableValue(c2_trainable, period=1000)
detailed_loss_tracker = SimpleLossTrackingCallback()
model = dde.Model(data_AC_2D_inverse, net)

LOSS_WEIGHTS = [1, 1, 1, 1, 1, 1, 1000]  # Weights for different components of the loss function

total_iterations = 0
while total_iterations < 100000:
                # Calculate the number of iterations for this loop
                iter_this_loop = 1000
                # Update the total iterations
                model.compile("adam", lr=1e-4, loss= 'MSE', loss_weights=LOSS_WEIGHTS, external_trainable_variables=[c2_trainable])

                losshistory, train_state = model.train(epochs=iter_this_loop, display_every=1000, callbacks=[variable, detailed_loss_tracker])
                # Update gamma value and error after training
                current_gamma_value = c2_trainable.value().numpy()

                # model.compile("L-BFGS", loss = 'MSE', loss_weights = Loss_Weights, external_trainable_variables=[gamma_2_AC])
                # losshistory, train_state = model.train(display_every=1000, callbacks=[observed_data_loss_callback, variable])

                # Update gamma value and error after training
                gamma_values.append(current_gamma_value)
                iterations_list.append(total_iterations + iter_this_loop)

                total_iterations += iter_this_loop

plt.figure(figsize=(10, 6))
plt.plot(iterations_list, gamma_values, '-o', label='Iteration vs Gamma Values', color='blue')
plt.xlabel('Gamma Value', fontsize=14)
plt.ylabel('Iterations', fontsize=14)
plt.title('Iterations vs. Gamma Value', fontsize=16)
plt.legend()
plt.grid(True, which="both", ls="--")
plt.show()


# Load the losses from the file
file_path = "/home/asfandyarkhan/PINN/data/losses_simple.txt"
losses_df = pd.read_csv(file_path)

# Calculate the total loss as the sum of component-wise losses for each iteration
# Assuming that the first column is 'Epoch' and the rest are loss components
loss_components = losses_df.columns[1:]  # Exclude 'Epoch'
losses_df['Total Loss'] = losses_df[loss_components].sum(axis=1)

# Plotting
plt.figure(figsize=(10, 6))

# Plot component-wise losses
for component in loss_components:
    plt.plot(losses_df['Epoch'], losses_df[component], label=component)

# Plot total loss
plt.plot(losses_df['Epoch'], losses_df['Total Loss'], label='Total Loss', color='black', linewidth=2, linestyle='--')

plt.xlabel('Iteration', fontsize=14)
plt.ylabel('Loss', fontsize=14)
plt.title('Loss Components and Total Loss over Iterations', fontsize=16)
plt.legend()
plt.grid(True)
plt.yscale('log')  # Use logarithmic scale if desired

plt.show()