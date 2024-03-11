import os
# Set backend
os.environ["DDE_BACKEND"] = "tensorflow"
#os.environ["DDE_BACKEND"] = "pytorch"
#os.environ["DDE_BACKEND"] = "paddle"

#import tensorflow as tf
import deepxde as dde
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from scipy.interpolate import interp1d
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.interpolate import RegularGridInterpolator

T_start = 0.0
T_step = 0.01
T_end = 0.1

# Define the computational domain
geom = dde.geometry.Interval(-1, 1)
timedomain = dde.geometry.TimeDomain(0, T_end)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)


# # Define gamma_2 as a trainable variable with an initial value
# #gamma_1 = dde.Variable(0.0001)
# gamma_2 = dde.Variable(0.1)
# # Allen-Cahn equation
# def allen_cahn(x, y):
#     dy_t = dde.grad.jacobian(y, x, i=0, j=1)
#     dy_xx = dde.grad.hessian(y, x, i=0, j=0)
#     gamma_1 = 0.0001  # Assume known
#     return dy_t - gamma_1 * dy_xx + gamma_2 * (y**3 - y)

gamma_2 = dde.Variable(1.0) # Heat equation 1D
def heat_1d(x, y):
    dy_t = dde.grad.jacobian(y, x, i=0, j=1)
    dy_xx = dde.grad.hessian(y, x, i=0, j=0)
    return dy_t - gamma_2 * dy_xx

# # Initial condition
# def init_condition(x):
#     return x[:, 0:1]**2 * np.sin(2 * np.pi * x[:, 0:1])

# Initial condition for heat equation 1D
def init_condition(x):
    return np.sin(np.pi * x[:, 0:1])
bc_heat = dde.icbc.DirichletBC(geomtime, lambda x: 0, lambda _, on_boundary: on_boundary)

initial_condition_h = dde.icbc.IC(geomtime, init_condition, lambda _, on_initial: on_initial, component=0)

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

# #print(observed_xt[1]), print(observed_h[1])
# # Data object for Allen Cahn Inverse
# data = dde.data.TimePDE(
#     geomtime,
#     allen_cahn,
#     [bc_h, bc_h_deriv, initial_condition_h, observe_h],  # Include observe_h here
#     num_domain=20000,
#     num_boundary=1600,
#     num_initial=4096,
#     anchors=observed_xt,  # Make sure observed_xt is used as anchors if necessary
#     #num_test=5000000,
# )

#print(observed_xt[1]), print(observed_h[1])
# Data object for Allen Cahn Inverse
# data = dde.data.TimePDE(
#     geomtime,
#     allen_cahn,
#     [bc_h, bc_h_deriv, initial_condition_h, observe_h],  # Include observe_h here
#     num_domain=20000,
#     num_boundary=1600,
#     num_initial=4096,
#     anchors=observed_xt,  # Make sure observed_xt is used as anchors if necessary
#     #num_test=5000000,
# )

data = dde.data.TimePDE(
    geomtime,
    heat_1d,
    [bc_heat, initial_condition_h, observe_h],
    num_domain=20000,
    num_boundary=1600,
    num_initial=4096,
    anchors=observed_xt,  # Make sure observed_xt is used as anchors if necessary
    num_test=500000,
)

# Neural network configuration
net = dde.nn.FNN([2] + [100] * 4 + [1], "tanh", "Glorot normal")
# Example of a more complex network
#net = dde.nn.FNN([2] + [128] * 6 + [1], "tanh", "Glorot normal")

# Model compilation
model = dde.Model(data, net)


BATCH_SIZE = 32  # Batch size
LEARNING_RATE = 1e-3  # Learning rate

ITERATIONS_A = 50000  # Number of training iterations
ITERATIONS_LBFGS = 50000  # Number of training iterations
ITERATIONS_A2 = 50000  # Number of training iterations
ITERATIONS_LBFGS2 = 50000  # Number of training iterations 

import tensorflow as tf



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

#LOSS_WEIGHTS = [100, 1, 1, 100, 100]  # Weights for different components of the loss function
#LOSS_WEIGHTS = [10, 1, 100, 1000]  # Weights for different components of the loss function


initial_learning_rate = 1e-3
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps=1000,
    decay_rate=0.96,
    staircase=True)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)


import tensorflow as tf

# Assuming observed_xt and observed_h are TensorFlow tensors or NumPy arrays
# that have been appropriately defined outside this function

def custom_loss(y_true, y_pred, model, observed_x, observed_t, observed_h):
    # Calculate PDE loss (as you normally would)
    pde_loss = tf.reduce_mean(tf.square(y_true - y_pred))
    
    # Prepare observed data for prediction
    observed_xt = tf.concat([observed_x, observed_t], axis=1)
    
    # Predict at observed points
    observed_pred = model(observed_xt, training=True)  # Ensure model is called in the correct mode
    
    # Calculate data fidelity loss
    data_loss = tf.reduce_mean(tf.square(observed_h - observed_pred))
    
    # Combine losses
    total_loss = pde_loss + data_loss  # You can introduce weights here as needed
    return total_loss

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

@tf.function
def train_step(model, inputs, outputs, observed_x, observed_t, observed_h):
    with tf.GradientTape() as tape:
        predictions = model(inputs, training=True)
        loss = custom_loss(outputs, predictions, model, observed_x, observed_t, observed_h)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss


# Assuming you've defined `custom_loss` as shown previously...
# And assuming `optimizer` is already defined, e.g.,
# optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

for epoch in range(100000):
    print(f"Epoch {epoch+1}/{100000}")
    
    # Shuffle your data or perform any necessary preprocessing
    # For simplicity, this step is not shown here
    
    for batch in 32:  # Assuming you've batched your data somehow
        inputs, outputs = batch  # Unpack your batched data
        
        # Perform a training step
        loss = train_step(model, inputs, outputs, observed_x, observed_t, observed_h)
    
    print(f"Loss at epoch {epoch+1}: {loss.numpy()}")
    
    # Optionally, validate your model on a validation set
    # Not shown here for simplicity


# Note: Ensure observed_xt and observed_h are prepared before this point
# For example, if they are NumPy arrays, convert them to TensorFlow tensors
observed_xt_tensor = tf.convert_to_tensor(observed_xt, dtype=tf.float32)
observed_h_tensor = tf.convert_to_tensor(observed_h, dtype=tf.float32)

# Prepare the custom loss function with tensors
custom_loss_function = custom_loss(model, observed_xt_tensor, observed_h_tensor, gamma_2, weight_pde=1, weight_data=100)

# Use the custom loss function when compiling the model
model.compile(optimizer="adam", lr=1e-3, loss=custom_loss_function)




#LOSS_WEIGHTS = [10, 1, 1, 1000]  # Weights for different components of the loss function
# Train on the current time step
#model.compile(optimizer, loss_weights=LOSS_WEIGHTS, external_trainable_variables=gamma_2)
#model.compile(optimizer, lr=LEARNING_RATE, external_trainable_variables=gamma_2)

#early_stopping = dde.callbacks.EarlyStopping(min_delta=1e-8, patience=5000)
#losshistory, train_state = model.train(iterations=80000, callbacks=[variable])
variable = dde.callbacks.VariableValue(gamma_2, period=1000)
#losshistory, train_state = model.train(iterations=ITERATIONS_A, batch_size=BATCH_SIZE, disregard_previous_best=True, callbacks=[early_stopping, variable])
losshistory, train_state = model.train(iterations=ITERATIONS_A, callbacks=[variable])

#dde.optimizers.config.set_LBFGS_options(maxcor=50, ftol=1e-15, gtol=1e-10, maxiter=ITERATIONS_LBFGS, maxfun=50000, maxls=50)
model.compile("L-BFGS-B", external_trainable_variables=gamma_2)
losshistory, train_state =model.train()

# #LOSS_WEIGHTS = [100, 1, 1, 100, 100]  # Weights for different components of the loss function
# LOSS_WEIGHTS2 = [10, 1, 1, 1000]  # Weights for different components of the loss function
# # Train on the current time step
# model.compile("adam", lr=LEARNING_RATE, loss_weights=LOSS_WEIGHTS2, external_trainable_variables=[gamma_2])
# #early_stopping = dde.callbacks.EarlyStopping(min_delta=1e-10, patience=10000)
# #losshistory, train_state = model.train(iterations=80000, callbacks=[variable])
# variable = dde.callbacks.VariableValue([gamma_2], period=1000)
# #losshistory, train_state = model.train(iterations=ITERATIONS_A, batch_size=BATCH_SIZE, disregard_previous_best=True, callbacks=[early_stopping, variable])
# losshistory, train_state = model.train(iterations=ITERATIONS_A, batch_size=BATCH_SIZE, callbacks=[early_stopping, variable])

# dde.optimizers.config.set_LBFGS_options(maxcor=50, ftol=1e-15, gtol=1e-10, maxiter=ITERATIONS_LBFGS, maxfun=50000, maxls=50)
# model.compile("L-BFGS-B", external_trainable_variables=[gamma_2])
# losshistory, train_state =model.train()


# After training, check the estimated value of gamma_2
print("Estimated gamma_2:", gamma_2.value().numpy())

predicted_h = model.predict(observed_xt)[:, 0]  # Predictions for observed points
# Calculate the mean absolute error between observed and predicted h
mae = np.mean(np.abs(predicted_h - observed_h))
print(f"MAE between observed and predicted h: {mae}")


LOSS_WEIGHTS = [10, 1, 1000]  # Weights for different components of the loss function

gamma_2_value = gamma_2.value().numpy()
def heat_1d_trained(x, y):
    dy_t = dde.grad.jacobian(y, x, i=0, j=1)
    dy_xx = dde.grad.hessian(y, x, i=0, j=0)
    return dy_t - gamma_2_value * dy_xx

# # Initial condition
# def init_condition(x):
#     return x[:, 0:1]**2 * np.sin(2 * np.pi * x[:, 0:1])

# Initial condition for heat equation 1D
def init_condition(x):
    return np.sin(np.pi * x[:, 0:1])
bc_heat = dde.icbc.DirichletBC(geomtime, lambda x: 0, lambda _, on_boundary: on_boundary)

initial_condition_h = dde.icbc.IC(geomtime, init_condition, lambda _, on_initial: on_initial, component=0)

data = dde.data.TimePDE(
    geomtime,
    heat_1d_trained,
    [bc_heat, initial_condition_h],
    num_domain=20000,
    num_boundary=1600,
    num_initial=4096,
)

# Neural network configuration
net = dde.nn.FNN([2] + [100] * 4 + [1], "tanh", "Glorot normal")
# Example of a more complex network
#net = dde.nn.FNN([2] + [128] * 6 + [1], "tanh", "Glorot normal")

# Model compilation
model_updated = dde.Model(data, net)

# Train on the current time step
model_updated.compile("adam", lr= 1e-3, loss_weights=LOSS_WEIGHTS)
#early_stopping = dde.callbacks.EarlyStopping(min_delta=1e-5, patience=5000)
#losshistory, train_state = model.train(iterations=80000, callbacks=[variable])
#losshistory, train_state = model.train(iterations=ITERATIONS_A, batch_size=BATCH_SIZE, disregard_previous_best=True, callbacks=[early_stopping, variable])
losshistory, train_state = model_updated.train(iterations=ITERATIONS_A)

#dde.optimizers.config.set_LBFGS_options(maxcor=50, ftol=1e-15, gtol=1e-10, maxiter=ITERATIONS_LBFGS, maxfun=50000, maxls=50)
model_updated.compile("L-BFGS-B")
losshistory, train_state =model_updated.train()

# predicted_h = model.predict(observed_xt)[:, 0]  # Predictions for observed points
# # Calculate the mean absolute error between observed and predicted h
# mae = np.mean(np.abs(predicted_h - observed_h))
# print(f"MAE between observed and predicted h: {mae}")




iteration = 1
Max_iterations = 0
error = 1  # Start with a high error
# Initialize a list to store gamma values and iterations
gamma_values = []
iterations_list = []
# Initialize a list to store errors over iterations
errors_over_iterations = []

#X = geomtime.random_points(100000)
X = geomtime.random_points(100000)  # Start with fewer points
while True:  # Loop indefinitely
        
        iteration += 1
        if iteration >= Max_iterations or error < 0.0001:
            break

        f = model.predict(X, operator=allen_cahn)
        # f is a list of arrays, extract predictions for 'h' and 'mu'
        #predictions_h = f[0].flatten()  # Flatten to make it 1D
        # Calculate absolute errors
        err_h = np.abs(f)
        # Calculate mean errors
        mean_error_h = np.mean(err_h)
        print(f"Iteration {iteration}, mean error: {mean_error_h}")
        # Strategies are 'mean_std' and 'quantile'
        threshold_h = dynamic_thresholding(err_h, strategy='mean_std')
        high_error_indices_h = np.where(err_h >= threshold_h)[0]

        if iteration >= 0 and len(high_error_indices_h) > 0:
            new_points_h = generate_diverse_points(X[high_error_indices_h], num_new_points=10, spread=0.01)
            X = np.vstack([X, new_points_h])
            print(f"Generated {len(new_points_h)} new points")

        # Train on the current time step
        model.compile("adam", lr=LEARNING_RATE, loss_weights=LOSS_WEIGHTS, external_trainable_variables=[gamma_2])
        early_stopping = dde.callbacks.EarlyStopping(min_delta=1e-10, patience=10000)
        losshistory, train_state = model.train(iterations=ITERATIONS_A2, batch_size=BATCH_SIZE, disregard_previous_best=True, callbacks=[early_stopping])

        dde.optimizers.config.set_LBFGS_options(maxcor=50, ftol=1e-15, gtol=1e-10, maxiter=ITERATIONS_LBFGS, maxfun=50000, maxls=50)
        model.compile("L-BFGS-B", external_trainable_variables=[gamma_2])
        losshistory, train_state = model.train()

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
        

# After training, 'gamma2.value' gives the estimated value of gamma2
# After training, check the estimated value of gamma_2
#print("Estimated gamma_1:", gamma_1.value().numpy())
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
x_range = np.linspace(-1, 1, 200+1).reshape(-1, 1)  # 400 points in x from -1 to 1
# Generate inputs for the two time steps
t0 = np.zeros_like(x_range)  # t = 0 for all x
t01 = np.full_like(x_range, 0.1)  # t = 0.1 for all x
# Stack x and t to form the input for the model
input_t0 = np.hstack((x_range, t0))
input_t01 = np.hstack((x_range, t01))
# Predict the solution at these two time steps

solution_t0 = model.predict(input_t0)  # Assuming the output is at index 0
solution_t01 = model.predict(input_t01)  # Assuming the output is at index 0

solution_t0_new = model_updated.predict(input_t0)  # Assuming the output is at index 0
solution_t01_new = model_updated.predict(input_t01)  # Assuming the output is at index 0



# Filter the rows for t=0 and t=0.1
indices_at_t0 = np.where(observed_t == 0.0)[0]
indices_at_t01 = np.where(observed_t == 0.1)[0]

data_x_at_t0 = observed_x[indices_at_t0]
data_h_at_t0 = observed_h[indices_at_t0]

data_x_at_t01 = observed_x[indices_at_t01]
data_h_at_t01 = observed_h[indices_at_t01]

# Stack x and t to form the input, though t is not used for the initial condition
#input_xt = np.hstack((x_range, 0.0))
# Evaluate the initial condition function for the x values
initial_condition_values = init_condition(input_t0)


# Now combine the plots
plt.figure(figsize=(10, 6))

# Plot the model predictions
plt.plot(x_range, solution_t0, label='Model Prediction at t=0', linestyle='--', color='blue')
plt.plot(x_range, solution_t01, label='Model Prediction at, t=0.1', linestyle='--', color='green')

plt.plot(x_range, solution_t0_new, label='Model Prediction at t=0 New One', linestyle='--', color='pink')
plt.plot(x_range, solution_t01_new, label='Model Prediction at t=0.1 New One', linestyle='--', color='black')

# Plot the observed data with solid lines
plt.plot(data_x_at_t0, data_h_at_t0, label='Observed Data at t=0', linestyle='-', color='red')
plt.plot(data_x_at_t01, data_h_at_t01, label='Observed Data at t=0.1', linestyle='-', color='orange')

plt.plot(x_range, initial_condition_values, label='Initial Condition at t=0 from formulae')


# Labeling the plot
plt.xlabel('x')
plt.ylabel('h/Solution')
plt.title('Comparison of Model Predictions and Observed Data at t=0 and t=0.1')
plt.legend()
plt.show()



# predicting the initial condition for the next time domain
def predict_at_time(model, time, nelx=200):
    x = np.linspace(-1, 1, nelx + 1).reshape(-1, 1)
    t = np.full_like(x, time)
    test_domain = np.hstack((x, t))
    return model.predict(test_domain)

def plot_model_predictions_over_time(model, start_time, end_time, time_step, nelx=200):
    # Spatial domain for plotting
    x = np.linspace(-1, 1, nelx + 1).reshape(-1, 1)
    # Set up the plot
    fig, ax = plt.subplots(figsize=(10, 7))
    line_styles = ['-', '--', '-.', ':']  # Different line styles for visual distinction
    # Generate predictions at each time step within the range and plot them
    times = np.arange(start_time, end_time + time_step, time_step)
    for i, time in enumerate(times):
        predictions = predict_at_time(model, time, nelx)
        h_predictions = predictions[:, 0]  # Assuming the first column contains 'h' predictions
        label = f't = {time:.2f}'
        ax.plot(x, h_predictions, line_styles[i % len(line_styles)], label=label)

    # Set labels and title
    ax.set_xlabel('x', fontsize=14)
    ax.set_ylabel('h(x, t)', fontsize=14)
    ax.set_title('Model Predictions at Various Time Steps', fontsize=16)
    # Add legend
    ax.legend(fontsize=12, loc='upper right')
    # Show plot
    plt.tight_layout()
    plt.show()

plot_model_predictions_over_time(model, T_start, T_step, T_end)