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

LOSS_WEIGHTS = [10, 1, 1, 1000]  # Weights for different components of the loss function
#LOSS_WEIGHTS = [10, 1, 1000]  # Weights for different components of the loss function
BATCH_SIZE = 32  # Batch size
LEARNING_RATE = 1e-3  # Learning rate

ITERATIONS_A = 20000  # Number of training iterations
ITERATIONS_LBFGS = 10000  # Number of training iterations
ITERATIONS_A2 = 20000  # Number of training iterations
ITERATIONS_LBFGS2 = 10000  # Number of training iterations 

# Define the computational domain
geom = dde.geometry.Interval(-1, 1)

# Define gamma_2 as a trainable variable with an initial value
gamma_2_AC = 4
# Allen-Cahn equation
def allen_cahn(x, y):
    dy_t = dde.grad.jacobian(y, x, i=0, j=1)
    dy_xx = dde.grad.hessian(y, x, i=0, j=0)
    gamma_1 = 0.0001  # Assume known
    return dy_t - gamma_1 * dy_xx + gamma_2_AC * (y**3 - y)

gamma_2_heat = 0.4  #0.4 # Heat equation 1D
def heat_1d(x, y):
    dy_t = dde.grad.jacobian(y, x, i=0, j=1)
    dy_xx = dde.grad.hessian(y, x, i=0, j=0)
    return dy_t - gamma_2_heat * dy_xx


# Initial condition Allen Cahn
def init_condition_AC(x):
    return x[:, 0:1]**2 * np.sin(2 * np.pi * x[:, 0:1])

# Initial condition for heat equation 1D
def init_condition_heat1D(x):
    return np.sin(np.pi * x[:, 0:1])

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

# Model training function to get weight and biases
def read_checkpoint_file(checkpoint_dir):
    checkpoint_file_path = os.path.join(checkpoint_dir, "checkpoint")
    try:
        with open(checkpoint_file_path, "r") as f:
            # Read all lines in the checkpoint file
            lines = f.readlines()
            # Extract the model checkpoint path
            for line in lines:
                if line.startswith("model_checkpoint_path"):
                    # Example line: model_checkpoint_path: "-2062.ckpt"
                    checkpoint_name = line.split(":")[1].strip().replace('"', '')
                    return checkpoint_name
    except FileNotFoundError:
        print(f"Checkpoint file not found in directory: {checkpoint_dir}")
        return None

# Model training function
models = []
# Initialize the network only once
# net = dde.nn.FNN([2] + [100] * 4 + [1], "tanh", "Glorot normal")
# model_start = dde.Model(data=None, net=net)  # We'll set data later
# Set backend and model save directory
save_dir_base = '/home/asfandyarkhan/PINN/weights/'
save_dir_base_data = '/home/asfandyarkhan/PINN/data/'


def train_model(data, current_time):

    if current_time == T_start:
        print("Initializing a new model for the first time step.")
        # Initialize your model architecture here
        net = dde.nn.FNN([2] + [100] * 4 + [1], "tanh", "Glorot normal")
        model = dde.Model(data, net)
    else:
        #model.data = data
        # Retrieve the latest checkpoint name for the previous time step
        checkpoint_name = read_checkpoint_file(save_dir_base)
        previous_model_filename = f"{checkpoint_name}"
        #previous_model_filename = f"{'none'}"
        previous_model_path = os.path.join(save_dir_base, previous_model_filename)

        print(f"Loading model from {previous_model_path}")
        print(previous_model_path)
        net = dde.nn.FNN([2] + [100] * 4 + [1], "tanh", "Glorot normal")
        model = dde.Model(data, net)
        model.restore(previous_model_path)
    
    # Train on the current time step
    model.compile("adam", lr=LEARNING_RATE)
    model.train(iterations=1000)
    model.compile("L-BFGS-B")
    model.train()

    iteration = 0
    Max_iterations = 2
    error = 1  # Start with a high error
    error_threshold = 0.0001  # Stop training if the error is less than this
    X = geomtime.random_points(10000)  # Start with fewer points
    while True:  # Loop indefinitel
            f = model.predict(X, operator=allen_cahn)
            # f is a list of arrays, extract predictions for 'h' and 'mu'
            #predictions_h = f[0].flatten()  # Flatten to make it 1D
            # Calculate absolute errors
            err_h = np.abs(f)
            # Calculate mean errors
            mean_error_h = np.mean(err_h)
            print(f"Iteration {iteration}, mean error: {mean_error_h}")

            # iteration and error check and model saving
            if iteration >= Max_iterations or error < error_threshold:
                # Ensure you are defining model_save_path here with the correct checkpoint_name
                model_save_path = os.path.join(save_dir_base, f"model_{current_time}_to_{next_time}")
                model.save(model_save_path)
                checkpoint_name = read_checkpoint_file(save_dir_base)
                print(checkpoint_name)
                if not checkpoint_name:
                    raise ValueError("Checkpoint name could not be retrieved.")
                break
            iteration += 1

            # Strategies are 'mean_std' and 'quantile'
            threshold_h = dynamic_thresholding(err_h, strategy='mean_std')
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
        ic_h_function = init_condition_AC
        #ic_h_function = init_condition_heat1D
    else:
        last_predictions = predict_at_time(models[-1], current_time, NELX)
        ic_h_function = create_initial_condition(last_predictions[:, 0], np.linspace(-1, 1, NELX + 1))

    # Heat 1D BC and ICs, and Data
    bc_heat = dde.icbc.DirichletBC(geomtime, lambda x: 0, lambda _, on_boundary: on_boundary) # Heat 1D
    initial_condition_h_heat1d = dde.icbc.IC(geomtime, init_condition_heat1D, lambda _, on_initial: on_initial, component=0) # Heat 1D
    initial_condition_h_heat1d = dde.icbc.IC(geomtime, init_condition_heat1D, lambda _, on_initial: on_initial, component=0)

    data_heat1D = dde.data.TimePDE(
        geomtime,
        heat_1d,
        [bc_heat, initial_condition_h_heat1d],
        num_domain=20000,
        num_boundary=1600,
        num_initial=4096,
        )
    
    # Allen Cahn BC and ICs and Data
    bc_h = dde.icbc.PeriodicBC(geomtime, 0, lambda _, on_boundary: on_boundary, derivative_order=0, component=0)
    bc_h_deriv = dde.icbc.PeriodicBC(geomtime, 0, lambda _, on_boundary: on_boundary, derivative_order=1, component=0)
    bc_mu = dde.icbc.PeriodicBC(geomtime, 0, lambda _, on_boundary: on_boundary, derivative_order=0, component=1)
    bc_mu_deriv = dde.icbc.PeriodicBC(geomtime, 0, lambda _, on_boundary: on_boundary, derivative_order=1, component=1)
    initial_condition_h_AC = dde.icbc.IC(geomtime, init_condition_AC, lambda _, on_initial: on_initial, component=0)

    data_AC = dde.data.TimePDE(
        geomtime,
        allen_cahn,
        [bc_h, bc_h_deriv, initial_condition_h_AC],  # Include observe_h here
        num_domain=20000,
        num_boundary=1600,
        num_initial=4096,
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


# def predict_and_save_h_over_time_range(models, start_time, end_time, time_step, nelx=100):
#     x = np.linspace(-1, 1, nelx)[:, None]  # Prepare spatial domain
#     times = np.linspace(start_time, end_time, num=2)  # Generates array [0.0, 0.25]
#     all_predictions = []

#     # # Ensure the number of models matches the number of time steps
#     # if len(models) != len(times):
#     #     print("Mismatch between number of models and time steps.")
#     #     return

#     for model, time in zip(models, times):
#         t = np.full_like(x, time)
#         xt = np.hstack((x, t))
#         predictions = model.predict(xt)
#         h_predictions = predictions[:, 0]  # Assuming the first column is 'h'
#         all_predictions.append(np.hstack((x, np.full_like(x, time), h_predictions[:, None])))

#     # Concatenate all predictions into a single array
#     all_predictions = np.vstack(all_predictions)

#     # Save to file
#     # Specify the directory where you want to save the file
#     save_directory = save_dir_base_data  # This should be the directory where you're saving your models
#     filename = "observed_h_data_over_time.csv"
#     full_path = os.path.join(save_directory, filename)
#     np.savetxt(full_path, all_predictions, delimiter=',', header='x,t,h', comments='')
#     print(f"Saved observed h data over time to {filename}")

# # Assuming you've stored each model trained for each time step in the `models` list
# predict_and_save_h_over_time_range(models, T_start, T_end, T_step, nelx=200)


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
    filename = "observed_h_data.csv"
    full_path = os.path.join(save_directory, filename)
    np.savetxt(full_path, all_predictions, delimiter=',', header='x,t,h', comments='')
    print(f"Saved observed h data over time to {full_path}")

# Assuming you've trained a model for the time range you're interested in, call the function with that model
predict_and_save_h_over_time_range(models[0], T_start, T_end, T_step, nelx=200)  # Replace 'single_model' with your actual model variable
