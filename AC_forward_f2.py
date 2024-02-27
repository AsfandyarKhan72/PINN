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


T_start = 0.0
T_step = 0.01
T_end = 0.1

# Define the computational domain
geom = dde.geometry.Interval(-1, 1)
timedomain = dde.geometry.TimeDomain(0, 0.1)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)


# Define gamma_2 as a trainable variable with an initial value
gamma_2 = 4
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

data = dde.data.TimePDE(
    geomtime,
    allen_cahn,
    [bc_h, bc_h_deriv, initial_condition_h],  # Include observe_h here
    num_domain=20000,
    num_boundary=1600,
    num_initial=4096,
)

# Neural network configuration
net = dde.nn.FNN([2] + [100] * 4 + [1], "tanh", "Glorot normal")
# Model compilation
model = dde.Model(data, net)

LOSS_WEIGHTS = [10, 1, 1, 1000]  # Weights for different components of the loss function
BATCH_SIZE = 32  # Batch size
LEARNING_RATE = 1e-3  # Learning rate

ITERATIONS_A = 20000  # Number of training iterations
ITERATIONS_LBFGS = 10000  # Number of training iterations
ITERATIONS_A2 = 20000  # Number of training iterations
ITERATIONS_LBFGS2 = 10000  # Number of training iterations 

# Train on the current time step
model.compile("adam", lr=LEARNING_RATE, loss_weights=LOSS_WEIGHTS)
early_stopping = dde.callbacks.EarlyStopping(min_delta=1e-10, patience=10000)
model.train(iterations=ITERATIONS_A, batch_size=BATCH_SIZE, disregard_previous_best=True, callbacks=[early_stopping])

dde.optimizers.config.set_LBFGS_options(maxcor=50, ftol=1e-15, gtol=1e-10, maxiter=ITERATIONS_LBFGS, maxfun=50000, maxls=50)
model.compile("L-BFGS-B")
losshistory, train_state =model.train()

# Sampling in sapce strategy
def generate_diverse_points(high_error_points, num_new_points, spread):
    perturbations = np.random.uniform(-spread, spread, size=(len(high_error_points), num_new_points, high_error_points.shape[1]))
    new_points = high_error_points[:, None, :] + perturbations
    return new_points.reshape(-1, high_error_points.shape[1])
# Sampling stragtegy to apply
def dynamic_thresholding(errors, strategy='mean_std'):
    if strategy == 'mean_std':
        threshold = np.mean(errors) + np.std(errors)
    elif strategy == 'quantile':
        threshold = np.quantile(errors, 0.9)
    else:
        raise ValueError("Unknown thresholding strategy")
    return threshold

iteration = 0
Max_iterations = 1
error = 1  # Start with a high error
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
        model.compile("adam", lr=LEARNING_RATE, loss_weights=LOSS_WEIGHTS)
        early_stopping = dde.callbacks.EarlyStopping(min_delta=1e-10, patience=10000)
        model.train(iterations=ITERATIONS_A, batch_size=BATCH_SIZE, disregard_previous_best=True, callbacks=[early_stopping])

        dde.optimizers.config.set_LBFGS_options(maxcor=50, ftol=1e-15, gtol=1e-10, maxiter=ITERATIONS_LBFGS, maxfun=50000, maxls=50)
        model.compile("L-BFGS-B")
        losshistory, train_state =model.train()

        iteration += 1
        if iteration >= Max_iterations or error < 0.0001:
            break

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

plot_model_predictions_over_time(model, 0, 0.1, 0.1)


def predict_and_save_h_over_time_range(model, start_time, end_time, time_step, filename="observed_h_data_over_time.csv", nelx=100):
    # Prepare arrays for spatial domain
    x = np.linspace(-1, 1, nelx)[:, None]
    # Generate time steps from start_time to end_time with a step of time_step
    times = np.arange(start_time, end_time + time_step, time_step)
    
    # Array to collect all predictions
    all_predictions = []

    for time_t in times:
        t = np.full_like(x, time_t)
        xt = np.hstack((x, t))
        predictions = model.predict(xt)
        h_predictions = predictions[:, 0]  # Assuming the first column is h
        all_predictions.append(np.hstack((x, np.full_like(x, time_t), h_predictions[:, None])))
    
    # Concatenate all predictions into a single array
    all_predictions = np.vstack(all_predictions)
    
    # Save to file
    np.savetxt(filename, all_predictions, delimiter=',', header='x,t,h', comments='')
    print(f"Saved observed h data over time to {filename}")

# Now call the function with time_step = 0.01
predict_and_save_h_over_time_range(model, T_start, T_end, T_step)


