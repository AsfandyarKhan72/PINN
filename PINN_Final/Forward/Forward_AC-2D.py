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
T_step = 0.25
T_end = 0.25

LOSS_WEIGHTS = [1, 1, 1, 1, 1, 1000]  # Weights for different components of the loss function
BATCH_SIZE = 32  # Batch size
LEARNING_RATE = 1e-3  # Learning rate

ITERATIONS_A = 50000  # Number of training iterations
ITERATIONS_LBFGS = 50000  # Number of training iterations
ITERATIONS_A2 = 50000  # Number of training iterations
ITERATIONS_LBFGS2 = 50000  # Number of training iterations 


# Define geometry and time domains
WIDTH = LENGTH = 1  # Domain size
geom = dde.geometry.Rectangle([0, 0], [WIDTH, LENGTH])  # Geometry domain

# Parameters
c1 = 0.0001  # Adjusted based on the paper
c2 = 4
def allen_cahn_2D (x, y):  # Allen Cahn 2D Mattey with single output 
    h = y
    h_t = dde.grad.jacobian(h, x, i=0, j=2)
    # Second derivatives of phi with respect to x and y
    phi_xx = dde.grad.hessian(h, x, i=0, j=0)
    phi_yy = dde.grad.hessian(h, x, i=1, j=1)
    phi_terms = phi_xx + phi_yy
    # Calculating the modified term for the Laplacian
    modified_phi_term = (c1 * phi_terms)- c2 * (h ** 3 - h)
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


# Model training function
models = []
save_dir_base = '/home/asfandyarkhan/PINN/weights/'
save_dir_base_data = '/home/asfandyarkhan/PINN/data/'

net = dde.nn.FNN([3] + [128] * 6 + [1], "tanh", "Glorot normal")
def train_model(data, current_time):
    # Define model
    model = dde.Model(data, net)
    iteration = 0
    Max_iterations = 3
    error = 1  # Start with a high error
    error_threshold = 0.0001  # Stop training if the error is less than this
    X = geomtime.random_points(10000)  # Start with fewer points
    while True:  # Loop indefinitel
            
            # Train on the current time step
            model.compile("adam", lr=LEARNING_RATE, loss_weights=LOSS_WEIGHTS)
            early_stopping = dde.callbacks.EarlyStopping(min_delta=1e-10, patience=10000)
            model.train(iterations=ITERATIONS_A, batch_size=BATCH_SIZE, disregard_previous_best=True, callbacks=[early_stopping])

            dde.optimizers.config.set_LBFGS_options(maxcor=50, ftol=1e-15, gtol=1e-10, maxiter=ITERATIONS_LBFGS, maxfun=50000, maxls=50)
            model.compile("L-BFGS-B")
            losshistory, train_state =model.train()

            f = model.predict(X, operator=allen_cahn_2D)
            err_h = np.abs(f)
            err_eq_flat = err_h.flatten()[:X.shape[0]]
            # Calculate mean errors
            mean_error_h = np.mean(err_eq_flat)
            print(f"Iteration {iteration}, mean error: {mean_error_h}")

            # iteration and error check and model saving
            if iteration >= Max_iterations or error < error_threshold:
                break
            iteration += 1

            # Strategies are 'mean_std' and 'quantile'
            threshold_h = dynamic_thresholding(err_h, strategy='mean_std')
            high_error_indices_h = np.where(err_h >= threshold_h)[0]

            if iteration >= 0 and len(high_error_indices_h) > 0:
                new_points_h = generate_diverse_points(X[high_error_indices_h], num_new_points=10, spread=0.01)
                X = np.vstack([X, new_points_h])
                print(f"Generated {len(new_points_h)} new points")

    return model

# Give predictions for AC 2D
def predict_at_time_2d(model, time, width=1.0, length=1.0, nelx=100, nely=100):
    # Create a grid in the spatial domain
    x = np.linspace(0, width, nelx)
    y = np.linspace(0, length, nely)
    # Create a constant array for the time dimension
    t = np.array([time])
    # Use meshgrid to create a 2D grid and then flatten it
    test_x, test_y, test_t = np.meshgrid(x, y, t)
    test_domain = np.vstack((test_x.flatten(), test_y.flatten(), test_t.flatten())).T
    # Predict using the model
    prediction = model.predict(test_domain)
    # Predict using the model
    return prediction

# Create IC for AC 2D
def create_initial_condition_2d(predictions, x_range, y_range):
    # Reshape predictions to a 2D grid
    predictions_reshaped = predictions.reshape(len(y_range), len(x_range))
    # Create a regular grid interpolator
    interp_func = RegularGridInterpolator((y_range, x_range), predictions_reshaped)
    # Return a function that applies this interpolation to new points
    return lambda X: interp_func(X[:, :2]).reshape(-1, 1)

NELX = 200

predictions = []
#models = []
# Predict at the first time step
x = np.linspace(0, WIDTH, 100)
y = np.linspace(0, LENGTH, 100)
# Equating the current time to the start time
current_time = T_start
while current_time < T_end:
    next_time = current_time + T_step
    # Prepare your data here
    time_domain = dde.geometry.TimeDomain(current_time, next_time)
    geomtime = dde.geometry.GeometryXTime(geom, time_domain)

    if current_time == T_start:
        ic_function = init_condition_AC_2D
    else:
        # Use the prediction from the previous time step
        previous_prediction = predictions[-1]
        ic_function = create_initial_condition_2d(previous_prediction, x, y)

    # Periodic Boundary Conditions for 'h' in x and y directions
    bc_h_x = dde.icbc.PeriodicBC(geomtime, component=0, derivative_order=0, component_x=0, on_boundary=on_boundary_x)
    bc_h_y = dde.icbc.PeriodicBC(geomtime, component=0, derivative_order=0, component_x=1, on_boundary=on_boundary_y)

    # Periodic Boundary Conditions for the derivative of 'h' in x and y directions
    bc_h_deriv_x = dde.icbc.PeriodicBC(geomtime, component=0, derivative_order=1, component_x=0, on_boundary=on_boundary_x)
    bc_h_deriv_y = dde.icbc.PeriodicBC(geomtime, component=0, derivative_order=1, component_x=1, on_boundary=on_boundary_y)

    # Initial Condition 
    initial_condition_AC_2d = dde.icbc.IC(geomtime, ic_function, lambda _, on_initial: on_initial)

    # Data for AC 2D
    data_AC_2D = dde.data.TimePDE(
        geomtime,
        allen_cahn_2D,
        [bc_h_x, bc_h_y, bc_h_deriv_x, bc_h_deriv_y, initial_condition_AC_2d],  # Include observe_h here
        num_domain=30000,
        num_boundary=1600,
        num_initial=4096,
    )
    # Train the model
    model_returned = train_model(data_AC_2D, current_time)
    models.append(model_returned)

    # Predict at the next time step
    t_start_prediction = predict_at_time_2d(model_returned, T_start, WIDTH, LENGTH, 100, 100)
    # Reshape prediction to align with the spatial domain
    t_start_prediction_reshaped = t_start_prediction[:, 0].reshape(100, 100)  # Assuming the first column is the desired output

    t_end_prediction = predict_at_time_2d(model_returned, T_end, WIDTH, LENGTH, 100, 100)
    t_start_prediction_reshaped = t_end_prediction[:, 0].reshape(100, 100)
    # Append the prediction to the list of predictions
    predictions.append(t_start_prediction_reshaped)
    predictions.append(t_start_prediction_reshaped)

    current_time = next_time

# Visualization - Plotting the contour for each time step
x = np.linspace(0, WIDTH, 100)
y = np.linspace(0, LENGTH, 100)
X, Y = np.meshgrid(x, y)

for i, prediction in enumerate(predictions):
    plt.figure()
    plt.contourf(X, Y, prediction, cmap="jet", levels=100)
    plt.colorbar()
    plt.title(f"Solution at t = {T_start + i * T_step}")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()

def predict_and_save_h_over_time_range_2d(model, start_time, end_time, time_step, nelx=100, nely=100, save_directory='/home/asfandyarkhan/PINN/data/'):
    x = np.linspace(0, WIDTH, nelx)
    y = np.linspace(0, LENGTH, nely)
    times = np.arange(start_time, end_time + time_step, time_step)
    all_predictions = []

    for time in times:
        X, Y = np.meshgrid(x, y)
        T = np.full_like(X, time)
        # Flatten X, Y, and T for model prediction
        test_domain = np.vstack([X.ravel(), Y.ravel(), T.ravel()]).T
        predictions = model.predict(test_domain).flatten()
        # Reshape predictions to nelx x nely for this example, adjust as needed
        predictions_reshaped = predictions.reshape(nelx, nely)
        for ix, iy in np.ndindex(predictions_reshaped.shape):
            all_predictions.append([x[ix], y[iy], time, predictions_reshaped[ix, iy]])

    # Save to file
    filename = "observed_AC_data_2d.csv"
    full_path = os.path.join(save_directory, filename)
    np.savetxt(full_path, all_predictions, delimiter=',', header='x,y,t,h', comments='')
    print(f"Saved observed h data over time to {full_path}")

# Assuming you've trained a model for the time range you're interested in, call the function with that model
predict_and_save_h_over_time_range_2d(models[0], T_start, T_end, 0.01, nelx=10, nely=10)
