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
from sklearn.metrics import mean_squared_error



T_start = 0.0
T_step = 0.25
T_end = 0.25

LOSS_WEIGHTS = [1, 1, 1, 1000, 1]  # Weights for different components of the loss function
#LOSS_WEIGHTS = [10, 1, 1000]  # Weights for different components of the loss function
BATCH_SIZE = 32  # Batch size
LEARNING_RATE = 1e-3  # Learning rate

ITERATIONS_A = 50000  # Number of training iterations
ITERATIONS_LBFGS = 20000  # Number of training iterations
ITERATIONS_A2 = 50000  # Number of training iterations
ITERATIONS_LBFGS2 = 10000  # Number of training iterations 

# Define the computational domain
geom = dde.geometry.Interval(-1, 1)

# Define gamma_2 as a trainable variable with an initial value
gamma_2_AC = dde.Variable(-4.0) # Heat equation 1D
# Allen-Cahn equation
def allen_cahn(x, y):
    dy_t = dde.grad.jacobian(y, x, i=0, j=1)
    dy_xx = dde.grad.hessian(y, x, i=0, j=0)
    gamma_1 = 0.0001  # Assume known
    return (dy_t - gamma_1 * dy_xx + gamma_2_AC * (y**3 - y))

gamma_2_heat = 0.4  #0.4 # Heat equation 1D
def heat_1d(x, y):
    dy_t = dde.grad.jacobian(y, x, i=0, j=1)
    dy_xx = dde.grad.hessian(y, x, i=0, j=0)
    return dy_t - gamma_2_heat * dy_xx


# Loading data for AC solution 
def load_observed_h(filename="/home/asfandyarkhan/PINN/data/observed_h_data.csv"):
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
observe_h_AC = dde.icbc.PointSetBC(observed_xt, observed_h, component=0)


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
save_dir_base = '/home/asfandyarkhan/PINN/weights/'
# Initialize a list to store gamma values and iterations
gamma_values = [gamma_2_AC.value().numpy()]  # Assuming this is how you access the value of your variable
iterations_list = [0]  # Starting with iteration 0
# Initialize a list to store errors over iterations
mean_errors_over_iterations = []
total_errors_over_iterations = []
update_interval = 1000  # Interval at which to update gamma and calculate error

def train_model(data, current_time, observed_xt, observed_h):

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
    if current_time == T_start:
        print("Training the model for the first time step.")
    else:
        model.compile("adam", lr=LEARNING_RATE)
        model.train(iterations=1000)
        model.compile("L-BFGS-B")
        model.train()

    iteration = 0
    Max_iterations = 1
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

                    # Initial setup before the loop
            total_iterations = 0

            #             # Assuming observed_xt and observed_h are available here
            # predicted_h_initial = model.predict(observed_xt)[:, 0]
            # initial_error = np.mean(np.abs(predicted_h_initial - observed_h))
            # errors_over_iterations.append(initial_error)

            # # # Now your lists are initialized as follows
            # # gamma_values = [initial_gamma_value]  # Already done in your existing code
            # # errors_over_iterations = [initial_error]  # Initialize with the initial error

            while total_iterations < ITERATIONS_A:
                # Calculate the number of iterations for this loop
                iter_this_loop = min(update_interval, ITERATIONS_A - total_iterations)
                
                # Training model
                model.compile("adam", lr=LEARNING_RATE, loss_weights=LOSS_WEIGHTS, external_trainable_variables=gamma_2_AC)
                early_stopping = dde.callbacks.EarlyStopping(min_delta=1e-10, patience=10000)
                model.train(iterations=iter_this_loop, batch_size=BATCH_SIZE, disregard_previous_best=True, callbacks=[early_stopping])

                # Update gamma value and error after training
                current_gamma_value = gamma_2_AC.value().numpy()
                gamma_values.append(current_gamma_value)
                iterations_list.append(total_iterations + iter_this_loop)

                # Inside your training loop, after training and updating gamma:
                predicted_h = model.predict(observed_xt)[:, 0]
                absolute_errors = np.abs(predicted_h - observed_h)
                mean_absolute_error = np.mean(absolute_errors)
                total_absolute_error = np.sum(absolute_errors)

                # Store both errors in their respective lists
                mean_errors_over_iterations.append(mean_absolute_error)
                total_errors_over_iterations.append(total_absolute_error)

                # Print current lengths of the lists
                print(f"After iteration {total_iterations + iter_this_loop}:")


                # Prepare for next loop
                total_iterations += iter_this_loop

            # model.compile("adam", lr=LEARNING_RATE, loss_weights=LOSS_WEIGHTS, external_trainable_variables=gamma_2_AC)
            # variable = dde.callbacks.VariableValue(gamma_2_AC, period=1000)
            # early_stopping = dde.callbacks.EarlyStopping(min_delta=1e-10, patience=10000)
            # model.train(iterations=ITERATIONS_A, batch_size=BATCH_SIZE, disregard_previous_best=True, callbacks=[early_stopping, variable])

            # # Inside your training loop, right after updating gamma_2_AC and printing its value:
            # gamma_value = gamma_2_AC.value().numpy()  # Assuming this is how you access the updated value
            # print("Current gamma value:", gamma_value)
            
            # # Store the updated gamma value and its corresponding iteration
            # gamma_values.append(gamma_value)
            # #iterations_list.append(iteration * 1000)  # Assuming you log every 1000 iterations
    
    
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
        [bc_h, bc_h_deriv, initial_condition_h_AC, observe_h_AC],  # Include observe_h here
        num_domain=20000,
        num_boundary=1600,
        num_initial=4096,
        anchors=observed_xt,  # Make sure observed_xt is used as anchors if necessary
        num_test=500000,
    )

    # Train the model
    model_returned = train_model(data_AC, current_time, observed_xt, observed_h)
    models.append(model_returned)

    current_time = next_time

# Assume gamma_values is your array of gamma values tracked during initial training
# Find the minimum error and its index
min_error = min(total_errors_over_iterations)
min_error_index = total_errors_over_iterations.index(min_error)

# Get the gamma value associated with the minimum error
gamma_with_min_error = gamma_values[min_error_index]

print(f"Minimum Error: {min_error}, at Gamma Value: {gamma_with_min_error}")

# If you want to reset gamma to the minimum value and continue training
gamma_2_AC.assign(gamma_with_min_error)  # Assuming gamma_2_AC is a tf.Variable or similar

New_LOSS_WEIGHTS = [10, 1, 1, 1000]  # Weights for different components of the loss function

net = dde.nn.FNN([2] + [100] * 4 + [1], "tanh", "Glorot normal")

data_AC = dde.data.TimePDE(
        geomtime,
        allen_cahn,
        [bc_h, bc_h_deriv, initial_condition_h_AC],  # Include observe_h here
        num_domain=20000,
        num_boundary=1600,
        num_initial=4096,
    )

model_new = dde.Model(data_AC, net)

iteration = 0
Max_iterations = 3
error = 1  # Start with a high error
error_threshold = 0.0001  # Stop training if the error is less than this
X = geomtime.random_points(10000)  # Start with fewer points
while True:  # Loop indefinitel
            f = model_new.predict(X, operator=allen_cahn)
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
                # model_save_path = os.path.join(save_dir_base, f"model_{current_time}_to_{next_time}")
                # model.save(model_save_path)
                # checkpoint_name = read_checkpoint_file(save_dir_base)
                # print(checkpoint_name)
                # if not checkpoint_name:
                #     raise ValueError("Checkpoint name could not be retrieved.")
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
            model_new.compile("adam", lr=LEARNING_RATE, loss_weights=New_LOSS_WEIGHTS)
            early_stopping = dde.callbacks.EarlyStopping(min_delta=1e-10, patience=10000)
            model_new.train(iterations=40000, batch_size=BATCH_SIZE, disregard_previous_best=True, callbacks=[early_stopping])

            dde.optimizers.config.set_LBFGS_options(maxcor=50, ftol=1e-15, gtol=1e-10, maxiter=20000, maxfun=50000, maxls=50)
            model_new.compile("L-BFGS-B")
            losshistory, train_state =model_new.train()

# # Train on the current time step
# model_new.compile("adam", lr=LEARNING_RATE, loss_weights=New_LOSS_WEIGHTS)
# early_stopping = dde.callbacks.EarlyStopping(min_delta=1e-10, patience=10000)
# model_new.train(iterations=50000, batch_size=BATCH_SIZE, disregard_previous_best=True, callbacks=[early_stopping])

# dde.optimizers.config.set_LBFGS_options(maxcor=50, ftol=1e-15, gtol=1e-10, maxiter=ITERATIONS_LBFGS, maxfun=50000, maxls=50)
# model_new.compile("L-BFGS-B")
# losshistory, train_state =model_new.train()



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

# Example: Assuming we're interested in time steps t=0 and t=0.25
t_interest = [0.0, 0.25]  # Time steps of interest

# Filter observed data for these specific time steps
observed_data_filtered = {
    t: {"x": observed_x[observed_t.flatten() == t], "h": observed_h[observed_t.flatten() == t]}
    for t in t_interest
}

# Your model predictions for t=0 and t=0.25
model_predictions = {
    t: model.predict(np.hstack((observed_data_filtered[t]["x"], np.full_like(observed_data_filtered[t]["x"], t))))
    for t in t_interest
}

# Define colors for plotting
colors = ['red', 'green', 'blue', 'orange']

# Your existing t_interest and observed_data_filtered setup...

# New model predictions for t=0 and t=0.25 using model_new
model_new_predictions = {
    t: model_new.predict(np.hstack((observed_data_filtered[t]["x"], np.full_like(observed_data_filtered[t]["x"], t))))
    for t in t_interest
}

# Plotting
plt.figure(figsize=(10, 6))

for idx, t in enumerate(t_interest):
    # Original observed data
    plt.plot(observed_data_filtered[t]["x"], observed_data_filtered[t]["h"], 'o', label=f'Observed Data at t={t}', color=colors[idx % len(colors)], linestyle='-', linewidth=2)
    
    # Original model predictions
    plt.plot(observed_data_filtered[t]["x"], model_predictions[t][:, 0], label=f'Original Model Prediction at t={t}', color=colors[(idx + 2) % len(colors)], linestyle='--', linewidth=2)
    
    # New model predictions
    plt.plot(observed_data_filtered[t]["x"], model_new_predictions[t][:, 0], label=f'Min Gamma Model Prediction at t={t}', color=colors[(idx + 3) % len(colors)], linestyle='-.', linewidth=2)

plt.xlabel('x', fontsize=14)
plt.ylabel('h(x, t)', fontsize=14)
plt.title('Comparison of Original and Min Gamma Model Predictions with Observed Data')
plt.legend()
plt.tight_layout()
plt.show()




plt.figure(figsize=(10, 6))
plt.plot(iterations_list, gamma_values, marker='o', linestyle='-', color='blue')
plt.xlabel('Iteration', fontsize=14)
plt.ylabel('Gamma Value', fontsize=14)
plt.title('Gamma Value Over Iterations', fontsize=16)
plt.grid(True)
plt.show()


if len(gamma_values) > len(mean_errors_over_iterations):
    gamma_values_corrected = gamma_values[1:]  # Skip initial gamma value
    iterations_list_corrected = iterations_list[1:]  # Skip initial iteration number
else:
    gamma_values_corrected = gamma_values
    iterations_list_corrected = iterations_list

# Ensure corrected lists are used for plotting
plt.figure(figsize=(10, 6))
plt.plot(iterations_list_corrected, mean_errors_over_iterations, '-o', label='Mean Absolute Error', color='blue')
plt.plot(iterations_list_corrected, total_errors_over_iterations, '-o', label='Total Absolute Error', color='red')
plt.xlabel('Iteration', fontsize=14)
plt.ylabel('Error', fontsize=14)
plt.title('Error Over Iterations', fontsize=16)
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))
if len(gamma_values_corrected) == len(mean_errors_over_iterations):
    plt.plot(gamma_values_corrected, mean_errors_over_iterations, '-o', label='Mean Absolute Error', color='blue')
    plt.plot(gamma_values_corrected, total_errors_over_iterations, '-o', label='Total Absolute Error', color='red')
    plt.xlabel('Gamma Value', fontsize=14)
    plt.ylabel('Error', fontsize=14)
    plt.title('Error vs. Gamma Value', fontsize=16)
    plt.legend()
    plt.grid(True)
    plt.show()
else:
    print("Mismatch in list sizes after correction:", len(gamma_values_corrected), len(mean_errors_over_iterations))




