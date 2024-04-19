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
# Define the computational domain
geom = dde.geometry.Interval(-1, 1)
time_domain = dde.geometry.TimeDomain(T_Start, T_End)
geomtime = dde.geometry.GeometryXTime(geom, time_domain)

# Define gamma_2 as a trainable variable with an initial value
gamma_1_AC = tf.Variable(1.0, dtype=tf.float32) # Heat equation 1D
gamma_2_AC = tf.Variable(1.0, dtype=tf.float32) # Heat equation 1D
def cahn_hilliard(x, y):
    dy_t = dde.grad.jacobian(y, x, i=0, j=1)
    laplacian_u = dde.grad.hessian(y, x, i=0, j=0)
    laplacian_u_cubed = dde.grad.hessian(y**3, x, i=0, j=0)   # second derivative y^3
    fourth_derivative_u = dde.grad.hessian(laplacian_u, x, i=0, j=0)
    return dy_t - gamma_2_AC * (laplacian_u_cubed - laplacian_u - gamma_1_AC * fourth_derivative_u)

# Initial condition Allen Cahn
def init_condition_CA(x):
    return -np.cos(2 * np.pi * x[:, 0:1])

# Initial condition dde for Allen Cahn Equation
initial_condition_h_AC = dde.icbc.IC(geomtime, init_condition_CA, lambda _, on_initial: on_initial, component=0)

# Boundary Condition for the Allen-Cahn equation
bc_h = dde.icbc.PeriodicBC(geomtime, 0, lambda _, on_boundary: on_boundary, derivative_order=0, component=0)
bc_h_deriv = dde.icbc.PeriodicBC(geomtime, 0, lambda _, on_boundary: on_boundary, derivative_order=1, component=0)


# Load the data from the text file
data = np.loadtxt('/home/asfandyarkhan/PINN/data/data_CA_1D.txt')
# Split the data into spatial coordinates (x), time (t), and solution values (u)
observed_x = data[:, 0:1]  # The x column (as 2D array)
observed_t = data[:, 1:2]  # The t column (as 2D array)
observed_h = data[:, 2:3]  # The solution column (as 2D array)
# Combine x and t to create the observation points in the space-time domain
observed_xt = np.hstack((observed_x, observed_t))
# Define the PointSetBC using the observed points and solution values
observe_h_AC = dde.icbc.PointSetBC(observed_xt, observed_h, component=0)

data_AC_inverse = dde.data.TimePDE(
        geomtime,
        cahn_hilliard,
        #[bc_h, bc_h_deriv, initial_condition_h_AC],  # Include observe_h here
        [bc_h, bc_h_deriv, initial_condition_h_AC, observe_h_AC],  # Include observe_h here
        num_domain=20000,
        num_boundary=1600,
        num_initial=4096,
        anchors=observed_xt,  # Make sure observed_xt is used as anchors if necessary
        num_test=40000,
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
                f.write("Epoch,PDE Loss,BC1 Loss,BC2 Loss,IC Loss,Observe Loss\n")

    def on_epoch_end(self):
        if self.model.train_state.step % self.every_n_epochs == 0 or self.model.train_state.step == 1:
            current_losses = self.model.train_state.loss_train
            loss_str = ",".join(map(str, current_losses))
            with open(self.file_path, "a") as f:
                f.write(f"{self.model.train_state.step},{loss_str}\n")

iterations_list = [0]  # Starting with iteration 0

gamma_1_values = [gamma_1_AC.value().numpy()]  # Assuming this is how you access the value of your variable
gamma_2_values = [gamma_2_AC.value().numpy()]  # Assuming this is how you access the value of your variable

# Network Architecure
#net = dde.nn.FNN([2] + [128] * 6 + [1], "tanh", "Glorot normal")
net = dde.nn.FNN([2] + [60] * 4 + [1], "tanh", "Glorot normal")
variable_gamma_1 = dde.callbacks.VariableValue(gamma_1_AC, period=1000)
variable_gamma_2 = dde.callbacks.VariableValue(gamma_2_AC, period=1000)
detailed_loss_tracker = SimpleLossTrackingCallback()
model = dde.Model(data_AC_inverse, net)

Loss_Weights = [1, 1, 1, 1, 1000]

total_iterations = 0
while total_iterations < 250000:
                # Calculate the number of iterations for this loop
                iter_this_loop = 1000
                # Update the total iterations
                #model.compile("adam", lr=1e-3, loss= 'MSE', loss_weights=Loss_Weights, external_trainable_variables=[gamma_1_AC, gamma_2_AC])
                #losshistory, train_state = model.train(epochs=70000, display_every=1000, callbacks=[variable_gamma_1, variable_gamma_2, detailed_loss_tracker])

                model.compile("adam", lr=1e-4, loss= 'MSE', loss_weights=Loss_Weights, external_trainable_variables=[gamma_1_AC, gamma_2_AC])
                losshistory, train_state = model.train(epochs=iter_this_loop, display_every=1000, callbacks=[variable_gamma_1, variable_gamma_2, detailed_loss_tracker])
                # Update gamma value and error after training
                current_gamma_1_value = gamma_1_AC.value().numpy()
                current_gamma_2_value = gamma_2_AC.value().numpy()

                # model.compile("L-BFGS", loss = 'MSE', loss_weights = Loss_Weights, external_trainable_variables=[gamma_2_AC])
                # losshistory, train_state = model.train(display_every=1000, callbacks=[observed_data_loss_callback, variable])

                # Update gamma value and error after training
                gamma_1_values.append(current_gamma_1_value)
                gamma_2_values.append(current_gamma_2_value)

                iterations_list.append(total_iterations + iter_this_loop)

                total_iterations += iter_this_loop

plt.figure(figsize=(10, 6))
plt.plot(iterations_list, gamma_1_values, '-o', label='Iteration vs Gamma Values', color='blue')
plt.plot(iterations_list, gamma_2_values, '-o', label='Iteration vs Gamma Values', color='red')
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