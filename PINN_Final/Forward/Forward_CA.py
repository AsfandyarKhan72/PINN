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
from pyDOE import lhs



T_start = 0.0
T_step = 0.25
T_end = 0.25

LOSS_WEIGHTS = [1, 1, 1, 1000]  # Weights for different components of the loss function
BATCH_SIZE = 32  # Batch size
LEARNING_RATE = 1e-3  # Learning rate

ITERATIONS_A = 40000  # Number of training iterations
ITERATIONS_LBFGS = 40000  # Number of training iterations
ITERATIONS_A2 = 40000  # Number of training iterations
ITERATIONS_LBFGS2 = 40000  # Number of training iterations 

# Define the computational domain
geom = dde.geometry.Interval(-1, 1)
# Define gamma_2 as a trainable variable with an initial value
D = 0.01
gamma = 0.0001
def cahn_hilliard(x, y):
    dy_t = dde.grad.jacobian(y, x, i=0, j=1)
    laplacian_u = dde.grad.hessian(y, x, i=0, j=0)
    laplacian_u_cubed = dde.grad.hessian(y**3, x, i=0, j=0)   # second derivative y^3
    fourth_derivative_u = dde.grad.hessian(laplacian_u, x, i=0, j=0)
    return dy_t - D * (laplacian_u_cubed - laplacian_u - gamma * fourth_derivative_u)

# Initial condition Allen Cahn
def init_condition_CA(x):
    return -np.cos(2 * np.pi * x[:, 0:1])

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

def visualize_points_1D(X, high_error_indices_h, new_points_h):
    plt.figure(figsize=(8, 6))
    # Plot original points before adding new ones in the current iteration
    # In 1D, we can plot these along the x-axis at a fixed y-value or just use the x-axis.
    plt.scatter(X[:, 0], np.zeros_like(X[:, 0]), c='lightgray', label='Original Points', alpha=0.5)
    # High error points from the last iteration
    high_error_points = X[high_error_indices_h]
    plt.scatter(high_error_points[:, 0], np.zeros_like(high_error_points[:, 0]), c='red', label='High Error Points', alpha=0.7)
    # Newly added points in the last iteration
    plt.scatter(new_points_h[:, 0], np.zeros_like(new_points_h[:, 0]), c='blue', label='New Points', alpha=1, edgecolor='black')
    plt.xlabel('X-axis')
    plt.yticks([])  # Remove y-axis ticks as they are not relevant in 1D
    plt.legend()
    plt.title('Adaptive Sampling Points in 1D')
    plt.show()


# Model training function
models = []
save_dir_base = '/home/asfandyarkhan/PINN/weights/'
save_dir_base_data = '/home/asfandyarkhan/PINN/data/'

#net = dde.nn.FNN([2] + [60] * 4 + [1], "tanh", "Glorot normal")
net = dde.nn.FNN([2] + [128] * 6 + [1], "tanh", "Glorot normal")
def train_model(data, current_time):
    # Define model
    model = dde.Model(data, net)
    iteration = 0
    Max_iterations = 5
    error = 1  # Start with a high error
    error_threshold = 0.0001  # Stop training if the error is less than this
    X = geomtime.random_points(100000)  # Start with fewer points
    while True:  # Loop indefinitel

            # Train on the current time step
            model.compile("adam", lr=LEARNING_RATE, loss_weights=LOSS_WEIGHTS)
            early_stopping = dde.callbacks.EarlyStopping(min_delta=1e-10, patience=10000)
            losshistory, train_state = model.train(iterations=ITERATIONS_A, batch_size=BATCH_SIZE, disregard_previous_best=True, callbacks=[early_stopping])

            dde.optimizers.config.set_LBFGS_options(maxcor=50, ftol=1e-15, gtol=1e-10, maxiter=ITERATIONS_LBFGS, maxfun=50000, maxls=50)
            model.compile("L-BFGS-B")
            losshistory, train_state = model.train()

            iteration += 1

            # Predict on the entire domain
            f = model.predict(X, operator=cahn_hilliard)
            err_h = np.abs(f)
            # Calculate mean errors
            mean_error_h = np.mean(err_h)
            print(f"Iteration {iteration}, mean error: {mean_error_h}")

            # Strategies are 'mean_std' and 'quantile'
            threshold_h = dynamic_thresholding(err_h, strategy='mean_std')
            print(f"Iteration {iteration}, threshold_h: {threshold_h}")

            high_error_indices_h = np.where(err_h >= threshold_h)[0]

            # Add new points exactly at high-error locations
            exact_high_error_points = X[high_error_indices_h]
            
            if iteration >= 0 and len(high_error_indices_h) > 0:
                nearby_new_points = generate_diverse_points(X[high_error_indices_h], num_new_points=1, spread=0.001)
                X = np.vstack([X, nearby_new_points])
                # Combine both sets of points
                #X = np.vstack([X, exact_high_error_points, nearby_new_points])
                print(f"Added {len(exact_high_error_points)} exact and {len(nearby_new_points)} nearby new points")
            
            # iteration and error check and model saving
            if iteration >= Max_iterations or error < error_threshold:
                break


            # Visualization
            visualize_points_1D(X, high_error_indices_h, nearby_new_points)

    return model

# predicting the initial condition for the next time domain
def predict_at_time(model, time, nelx=200):
    x = np.linspace(-1, 1, nelx + 1).reshape(-1, 1)
    t = np.full_like(x, time)
    test_domain = np.hstack((x, t))
    return model.predict(test_domain)

# Utility Functions
def create_initial_condition(predictions, x_range, kind='linear'):
    #h_predictions = predictions[:, 0]
    h_predictions = predictions
    interp_func = interp1d(x_range, h_predictions.flatten(), kind=kind, fill_value="extrapolate")
    return lambda x: interp_func(x[:, 0]).reshape(-1, 1)

NELX = 200
current_time = T_start
predictions = []
#models = []
while current_time < T_end:
    next_time = current_time + T_step
    # Prepare your data here
    time_domain = dde.geometry.TimeDomain(current_time, next_time)
    geomtime = dde.geometry.GeometryXTime(geom, time_domain)

    if current_time == T_start:
        ic_h_function = init_condition_CA
        #ic_h_function = init_condition_heat1D
    else:
        last_predictions = predict_at_time(models[-1], current_time, NELX)
        ic_h_function = create_initial_condition(last_predictions[:, 0], np.linspace(-1, 1, NELX + 1))

    # Allen Cahn BC and ICs and Data
    bc_h = dde.icbc.PeriodicBC(geomtime, 0, lambda _, on_boundary: on_boundary, derivative_order=0, component=0)
    bc_h_deriv = dde.icbc.PeriodicBC(geomtime, 0, lambda _, on_boundary: on_boundary, derivative_order=1, component=0)
    # Initial Condition 
    initial_condition_h_AC = dde.icbc.IC(geomtime, init_condition_CA, lambda _, on_initial: on_initial, component=0)

    data_AC = dde.data.TimePDE(
        geomtime,
        cahn_hilliard,
        [bc_h, bc_h_deriv, initial_condition_h_AC],  # Include observe_h here
        num_domain=10000,
        num_boundary=200,
        num_initial=512,
    )

    # Train the model
    model_returned = train_model(data_AC, current_time)
    models.append(model_returned)
    #current_time = next_time

    # Assuming 'models' is a list of models where each model corresponds to a specific time step

    # Spatial domain for plotting
    x = np.linspace(-1, 1, NELX + 1).reshape(-1, 1)
    # Plot setup
    fig, ax = plt.subplots(figsize=(10, 7))
    line_styles = ['-', '--', '-.', ':']  # Different line styles for visual distinction
    # Time steps for prediction
    time_steps = np.linspace(current_time, next_time, 1)
    #time_steps = np.current_time

    # Iterate through each model and corresponding time step
    for i, (model, time_step) in enumerate(zip(models, time_steps)):
        # Predict and plot for this time step
        predictions = predict_at_time(model, time_step, NELX)
        h_predictions = predictions[:, 0]
        label = f't = {time_step:.2f}'
        ax.plot(x, h_predictions, line_styles[i % len(line_styles)], label=label)

    # Labeling and showing plot
    ax.set_xlabel('x', fontsize=14)
    ax.set_ylabel('h(x, t)', fontsize=14)
    ax.set_title('Model Predictions at Various Time Steps', fontsize=16)
    ax.legend(fontsize=12, loc='upper right')
    plt.tight_layout()
    plt.show()

    current_time = next_time

# Spatial domain for plotting
x = np.linspace(-1, 1, NELX + 1).reshape(-1, 1)

# Set up the plot
fig, ax = plt.subplots(figsize=(10, 7))
line_styles = ['-', '--', '-.', ':']  # Different line styles for visual distinction

# Iterate through each model to plot predictions for each time step
for i, model in enumerate(models):
    # Start of the current model's time domain
    start_time = T_start + i * T_step
    # End of the current model's time domain
    end_time = start_time + T_step if i < len(models) - 1 else T_end

    # Predict and plot for the start time
    predictions_start = predict_at_time(model, start_time, NELX)
    h_predictions_start = predictions_start[:, 0]
    label_start = f't = {start_time:.2f}'
    ax.plot(x, h_predictions_start, line_styles[i % len(line_styles)], label=label_start)

    # For all models, also predict and plot for their end time
    if end_time != start_time:  # This check is technically redundant now
        predictions_end = predict_at_time(model, end_time, NELX)
        h_predictions_end = predictions_end[:, 0]
        label_end = f't = {end_time:.3f}'
        # Using 'dotted' linestyle for end time predictions for clarity
        ax.plot(x, h_predictions_end, line_styles[i % len(line_styles)], linestyle='dotted', label=label_end)

# Set labels and title
ax.set_xlabel('x', fontsize=14)
ax.set_ylabel('h(x, t)', fontsize=14)
ax.set_title('Model Predictions at Various Time Steps', fontsize=16)

# Add legend
ax.legend(fontsize=12, loc='upper right')

# Show plot
plt.tight_layout()
plt.show()

# Save to file
def predict_and_save_h_over_time_range(model, start_time, end_time, time_step, nelx=100, save_directory='/home/asfandyarkhan/PINN/data/'):
    x = np.linspace(-1, 1, nelx)[:, None]  # Prepare spatial domain
    times = np.arange(start_time, end_time + time_step, time_step)  # Time steps including the end time
    all_predictions = []

    for time in times:
        t = np.full_like(x, time)
        xt = np.hstack((x, t))
        predictions = model.predict(xt)
        h_predictions = predictions[:, 0]  # Assuming the first column is 'h'
        all_predictions.append(np.hstack((x, t, h_predictions[:, None])))

    # Concatenate all predictions into a single array
    all_predictions = np.vstack(all_predictions)

    # Save to file
    filename = "observed_h_data_CA.csv"
    full_path = os.path.join(save_directory, filename)
    np.savetxt(full_path, all_predictions, delimiter=',', header='x,t,h', comments='')
    print(f"Saved observed h data over time to {full_path}")

# Assuming you've trained a model for the time range you're interested in, call the function with that model
predict_and_save_h_over_time_range(models[0], 0, 0.25, 0.01, nelx=201)  # Replace 'single_model' with your actual model variable