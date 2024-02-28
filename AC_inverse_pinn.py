import os
# Set backend
os.environ["DDE_BACKEND"] = "tensorflow"
import tensorflow as tf
import deepxde as dde
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.interpolate import RegularGridInterpolator

# Define the computational domain
geom = dde.geometry.Interval(-1, 1)
timedomain = dde.geometry.TimeDomain(0, 0.1)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)


# Define gamma_2 as a trainable variable with an initial value
gamma_2 = dde.Variable(4.0)
# Allen-Cahn equation
def allen_cahn(x, y):
    dy_t = dde.grad.jacobian(y, x, i=0, j=1)
    dy_xx = dde.grad.hessian(y, x, i=0, j=0)
    gamma_1 = 0.0001  # Assume known
    return dy_t - gamma_1 * dy_xx + gamma_2 * (y**3 - y)

# Initial condition
def init_condition(x):
    return x[:, 0:1]**2 * np.sin(2 * np.pi * x[:, 0:1])

initial_condition_h = dde.icbc.IC(geomtime, init_condition, lambda _, on_initial: on_initial, component=0)
bc_h = dde.icbc.PeriodicBC(geomtime, 0, lambda _, on_boundary: on_boundary, derivative_order=0, component=0)
bc_h_deriv = dde.icbc.PeriodicBC(geomtime, 0, lambda _, on_boundary: on_boundary, derivative_order=1, component=0)


# Loading data for AC solution 
def load_observed_h(filename="/home/asfandyarkhan/Documents/observed_h_data_over_time.csv"):
    # Load the saved data
    data = np.loadtxt(filename, delimiter=',', skiprows=1)
    # Extract x, t, and observed h values
    observed_x = data[:, 0:1]
    observed_t = data[:, 1:2]
    observed_h = data[:, 2]

    return observed_x, observed_t, observed_h

# Load the observed data
observed_x, observed_t, observed_h = load_observed_h()
# Assuming you're setting up your inverse problem here
# Create the [x, t] pairs for the observed data
observed_xt = np.hstack((observed_x, observed_t))
# Use observed_h in your PointSetBC or as part of your data for the inverse problem
observe_h = dde.icbc.PointSetBC(observed_xt, observed_h, component=0)

# Data object for Allen Cahn Inverse
data = dde.data.TimePDE(
    geomtime,
    allen_cahn,
    [bc_h, bc_h_deriv, initial_condition_h, observe_h],  # Include observe_h here
    num_domain=20000,
    num_boundary=1600,
    num_initial=4096,
    anchors=observed_xt,  # Make sure observed_xt is used as anchors if necessary
    #num_test=10000,
)


# Neural network configuration
net = dde.nn.FNN([2] + [100] * 4 + [1], "tanh", "Glorot normal")
# Model compilation
model = dde.Model(data, net)

LOSS_WEIGHTS = [10, 1, 1, 10, 1000]  # Weights for different components of the loss function
BATCH_SIZE = 32  # Batch size
LEARNING_RATE = 1e-4  # Learning rate

ITERATIONS_A = 50000  # Number of training iterations
ITERATIONS_LBFGS = 50000  # Number of training iterations
ITERATIONS_A2 = 50000  # Number of training iterations
ITERATIONS_LBFGS2 = 50000  # Number of training iterations 

# Sampling stretegy
def generate_diverse_points(high_error_points, num_new_points, spread):
    perturbations = np.random.uniform(-spread, spread, size=(len(high_error_points), num_new_points, high_error_points.shape[1]))
    new_points = high_error_points[:, None, :] + perturbations
    return new_points.reshape(-1, high_error_points.shape[1])
# Sampling Strategy 
def dynamic_thresholding(errors, strategy='mean_std'):
    if strategy == 'mean_std':
        threshold = np.mean(errors) + np.std(errors)
    elif strategy == 'quantile':
        threshold = np.quantile(errors, 0.9)
    else:
        raise ValueError("Unknown thresholding strategy")
    return threshold

# Train on the current time step
model.compile("adam", lr=LEARNING_RATE, loss_weights=LOSS_WEIGHTS, external_trainable_variables=[gamma_2])
early_stopping = dde.callbacks.EarlyStopping(min_delta=1e-10, patience=10000)
model.train(iterations=ITERATIONS_A, batch_size=BATCH_SIZE, disregard_previous_best=True, callbacks=[early_stopping])

dde.optimizers.config.set_LBFGS_options(maxcor=50, ftol=1e-15, gtol=1e-10, maxiter=50000, maxfun=50000, maxls=50)
model.compile("L-BFGS-B", external_trainable_variables=[gamma_2])
losshistory, train_state =model.train()

# After training, check the estimated value of gamma_2
print("Estimated gamma_2:", gamma_2.value().numpy())

iteration = 1
Max_iterations = 5
error = 1  # Start with a high error
# Initialize a list to store gamma values and iterations
gamma_values = []
iterations_list = []
# Initialize a list to store errors over iterations
errors_over_iterations = []

#X = geomtime.random_points(100000)
X = geomtime.random_points(10000)  # Start with fewer points
while True:  # Loop indefinitely
        f = model.predict(X, operator=allen_cahn)
        # f is a list of arrays, extract predictions for 'h' and 'mu'
        predictions_h = f[0].flatten()  # Flatten to make it 1D
        # Calculate absolute errors
        err_h = np.abs(predictions_h)
        # Calculate mean errors
        mean_error_h = np.mean(err_h)
        print(f"Iteration {iteration}, mean error: {mean_error_h}")
        # Strategies are 'mean_std' and 'quantile'
        threshold_h = dynamic_thresholding(err_h, strategy='quantile')
        high_error_indices_h = np.where(err_h >= threshold_h)[0]

        if iteration >= 0 and len(high_error_indices_h) > 0:
            new_points_h = generate_diverse_points(X[high_error_indices_h], num_new_points=10, spread=0.01)
            X = np.vstack([X, new_points_h])
            print(f"Generated {len(new_points_h)} new points")

        # Train on the current time step
        model.compile("adam", lr=LEARNING_RATE, loss_weights=LOSS_WEIGHTS, external_trainable_variables=[gamma_2])
        early_stopping = dde.callbacks.EarlyStopping(min_delta=1e-10, patience=10000)
        model.train(iterations=ITERATIONS_A2, batch_size=BATCH_SIZE, disregard_previous_best=True, callbacks=[early_stopping])

        dde.optimizers.config.set_LBFGS_options(maxcor=50, ftol=1e-15, gtol=1e-10, maxiter=ITERATIONS_LBFGS, maxfun=50000, maxls=50)
        model.compile("L-BFGS-B", external_trainable_variables=[gamma_2])
        losshistory, train_state =model.train()

        # Record the current value of gamma_2 and the iteration
        gamma_value = gamma_2.value().numpy()
        print("Gamma value:", gamma_value)
        gamma_values.append(gamma_value)
        iterations_list.append(iteration*10000)

        # After training, calculate predictions for the observed data
        predicted_h = model.predict(observed_xt)[:, 0]  # Assuming this gives us the predictions for observed points
        # Calculate the mean absolute error (MAE) for this iteration
        mae = np.mean(np.abs(predicted_h - observed_h))
        errors_over_iterations.append(mae)
        print(f"Iteration {iteration}, MAE: {mae}")
        
        iteration += 1
        if iteration >= Max_iterations or error < 0.0001:
            break

# After training, 'gamma2.value' gives the estimated value of gamma2
# After training, check the estimated value of gamma_2
print("Estimated gamma_2:", gamma_2.value().numpy())


# Plotting gamma values against iterations
plt.figure(figsize=(10, 6))
plt.plot(iterations_list, gamma_values, marker='o', linestyle='-')
plt.xlabel('Iteration')
plt.ylabel('Gamma Value')
plt.title('Gamma Value Over Iterations')
plt.grid(True)
plt.show()

# Plotting the errors over iterations
plt.figure(figsize=(10, 6))
plt.plot(iterations_list, errors_over_iterations, marker='o', linestyle='-')
plt.xlabel('Iteration')
plt.ylabel('Mean Absolute Error (MAE)')
plt.title('Error Over Iterations')
plt.grid(True)
plt.show()


# Define the x range
x_range = np.linspace(-1, 1, 400).reshape(-1, 1)  # 400 points in x from -1 to 1
# Generate inputs for the two time steps
t0 = np.zeros_like(x_range)  # t = 0 for all x
t01 = np.full_like(x_range, 0.1)  # t = 0.1 for all x
# Stack x and t to form the input for the model
input_t0 = np.hstack((x_range, t0))
input_t01 = np.hstack((x_range, t01))
# Predict the solution at these two time steps
solution_t0 = model.predict(input_t0)[:, 0]  # Assuming the output is at index 0
solution_t01 = model.predict(input_t01)[:, 0]  # Assuming the output is at index 0

# Filter the rows for t=0 and t=0.1
indices_at_t0 = np.where(observed_t == 0.0)[0]
indices_at_t01 = np.where(observed_t == 0.1)[0]

data_x_at_t0 = observed_x[indices_at_t0]
data_h_at_t0 = observed_h[indices_at_t0]

data_x_at_t01 = observed_x[indices_at_t01]
data_h_at_t01 = observed_h[indices_at_t01]


# Now combine the plots
plt.figure(figsize=(10, 6))

# Plot the model predictions
plt.plot(x_range, solution_t0, label='Model Prediction at t=0', linestyle='--', color='blue')
plt.plot(x_range, solution_t01, label='Model Prediction at t=0.1', linestyle='--', color='green')

# Plot the observed data with solid lines
plt.plot(data_x_at_t0, data_h_at_t0, label='Observed Data at t=0', linestyle='-', color='red')
plt.plot(data_x_at_t01, data_h_at_t01, label='Observed Data at t=0.1', linestyle='-', color='orange')

# Labeling the plot
plt.xlabel('x')
plt.ylabel('h/Solution')
plt.title('Comparison of Model Predictions and Observed Data at t=0 and t=0.1')
plt.legend()
plt.show()

