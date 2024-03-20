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

ITERATIONS_A = 15000  # Number of training iterations
ITERATIONS_LBFGS = 20000  # Number of training iterations
ITERATIONS_A2 = 20000  # Number of training iterations
ITERATIONS_LBFGS2 = 10000  # Number of training iterations 

# Define the computational domain
geom = dde.geometry.Interval(-1, 1)

# Define gamma_2 as a trainable variable with an initial value
gamma_2_AC = dde.Variable(3.0) # Heat equation 1D
# Allen-Cahn equation
def allen_cahn(x, y):
    dy_t = dde.grad.jacobian(y, x, i=0, j=1)
    dy_xx = dde.grad.hessian(y, x, i=0, j=0)
    gamma_1 = 0.0001  # Assume known
    #return (dy_t - gamma_2_AC * dy_xx + 4 * (y**3 - y))
    return (dy_t - gamma_1 * dy_xx + gamma_2_AC * (y**3 - y))

gamma_2_heat = 0.4  #0.4 # Heat equation 1D
def heat_1d(x, y):
    dy_t = dde.grad.jacobian(y, x, i=0, j=1)
    dy_xx = dde.grad.hessian(y, x, i=0, j=0)
    return dy_t - gamma_2_heat * dy_xx


# Loading data for AC solution 
def load_observed_h(filename="/home/asfandyarkhan/PINN/data/observed_h_data_copy_1.csv"):
    # Load the saved data
    data = np.loadtxt(filename, delimiter=',', skiprows=1)
    # Extract x, t, and observed h values
    observed_x = data[:, 0:1]
    observed_t = data[:, 1:2]
    observed_h = data[:, 2]
    #print(observed_x[0], observed_t[0], observed_h[0])

    return observed_x, observed_t, observed_h

# Load the observed data
observed_x, observed_t, observed_h = load_observed_h()
# Assuming you're setting up your inverse problem here
# Create the [x, t] pairs for the observed data
observed_xt = np.hstack((observed_x, observed_t))


# Use observed_h in your PointSetBC or as part of your data for the inverse problem
observe_h_AC = dde.icbc.PointSetBC(observed_xt, observed_h)


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


# Your file path
file_path = "/home/asfandyarkhan/PINN/data/losses_simple.txt"
# Check if file exists and delete it
if os.path.exists(file_path):
    os.remove(file_path)
    print(f"Removed existing file: {file_path}")

from deepxde.callbacks import Callback

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

gamma_adjustment_factor = 3  # Factor to adjust gamma
error_check_frequency = 3  # Check error trend every 'n' iterations
total_errors_over_iterations2 = []  # List to store the total errors over iterations
def adjust_gamma_based_on_error(gamma_variable, current_total_error):
    global total_errors_over_iterations2

    # Append the current total error to the list
    total_errors_over_iterations2.append(current_total_error)

    # Ensure we have enough errors to compare
    print(f"length of total_errors_over_iterations: {len(total_errors_over_iterations2)}")
    if len(total_errors_over_iterations2) >= error_check_frequency:
        # Get the last three errors
        last_three_errors = total_errors_over_iterations2[-error_check_frequency:]
        print(f"last_three_errors: {last_three_errors}")

        # Check if the most recent error is less than both previous errors
        if last_three_errors[2] < last_three_errors[0] and last_three_errors[2] < last_three_errors[1]:
            print("No adjustment needed as error is decreasing.")
        else:
            # Error is not decreasing; adjust gamma
            new_gamma_value = gamma_variable.value().numpy() * -gamma_adjustment_factor
            gamma_variable.assign(new_gamma_value)
            print(f"Gamma adjusted to {new_gamma_value}")

        # Remove the oldest error to continue the process for the next set of errors
        total_errors_over_iterations2.pop(0)


import tensorflow as tf
import deepxde as dde

class CustomModel(dde.Model):
    def __init__(self, data, net, observed_xt, observed_h, observed_weight=100.0):
        super(CustomModel, self).__init__(data, net)
        # Ensure observed_xt and observed_h are tensors
        self.observed_xt = tf.convert_to_tensor(observed_xt, dtype=tf.float32)
        self.observed_h = tf.convert_to_tensor(observed_h, dtype=tf.float32)
        self.observed_weight = observed_weight

    def train_step(self, X, y, **kwargs):
        # Perform the standard training step
        loss_value, gradients = super(CustomModel, self).train_step(X, y, **kwargs)
        
        # Add observed data loss
        with tf.GradientTape() as tape:
            tape.watch(self.net.trainable_variables)
            predicted_h = self.net(self.observed_xt, training=True)
            observed_loss = tf.reduce_mean(tf.square(predicted_h - self.observed_h))
        
        observed_gradients = tape.gradient(observed_loss, self.net.trainable_variables)
        
        # Combine gradients and apply updates
        combined_gradients = [g1 + self.observed_weight * g2 for g1, g2 in zip(gradients, observed_gradients)]
        self.optimizer.apply_gradients(zip(combined_gradients, self.net.trainable_variables))
        
        # Return the combined loss value and gradients
        return loss_value + self.observed_weight * observed_loss, combined_gradients

    def total_loss(self, targets, outputs, loss_fn, inputs):
        # Calculate standard losses (PDEs, BCs, ICs)
        standard_loss = super(CustomModel, self).total_loss(targets, outputs, loss_fn, inputs)
        
        # Calculate observed data loss
        predicted_h = self.net(self.observed_xt, training=False)
        observed_loss = tf.reduce_mean(tf.square(predicted_h - self.observed_h))
        
        # Combine losses
        total_loss = standard_loss + self.observed_weight * observed_loss
        return total_loss





    
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

    global gamma_2_AC

    if current_time == T_start:
        print("Initializing a new model for the first time step.")
        # Initialize your model architecture here
        net = dde.nn.FNN([2] + [100] * 4 + [1], "tanh", "Glorot normal")
        #model = dde.Model(data, net)
        model = CustomModel(data, net, observed_xt, observed_h)
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
    
    # model.compile("adam", lr=LEARNING_RATE)
    # model.train(iterations=10000, batch_size=BATCH_SIZE)
    # model.compile("L-BFGS-B")
    # model.train()

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

            while total_iterations < ITERATIONS_A:
                # Calculate the number of iterations for this loop
                #iter_this_loop = min(update_interval, ITERATIONS_A - total_iterations)
                iter_this_loop = 1000

                # initial_learning_rate = 1e-3
                # lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                #     initial_learning_rate,
                #     decay_steps=1000,
                #     decay_rate=0.9,
                #     staircase=True)

                # optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
                
                # Training model
                model.compile("adam", lr=LEARNING_RATE, loss_weights=LOSS_WEIGHTS, external_trainable_variables=gamma_2_AC)
                #model.compile(optimizer,loss_weights=LOSS_WEIGHTS, external_trainable_variables=gamma_2_AC)
                early_stopping = dde.callbacks.EarlyStopping(min_delta=1e-10, patience=10000)
                # Initialize the detailed loss tracking callback
                detailed_loss_tracker = SimpleLossTrackingCallback()
                losshistory, train_state = model.train(iterations=iter_this_loop, batch_size=BATCH_SIZE, disregard_previous_best=True, callbacks=[early_stopping, detailed_loss_tracker])

                # dde.optimizers.config.set_LBFGS_options(maxcor=50, ftol=1e-15, gtol=1e-10, maxiter=ITERATIONS_LBFGS, maxfun=50000, maxls=50)
                # model.compile("L-BFGS-B", external_trainable_variables=gamma_2_AC)
                # losshistory, train_state =model.train()

                # Update gamma value and error after training
                current_gamma_value = gamma_2_AC.value().numpy()
                print(f"Gamma value after {total_iterations + iter_this_loop} iterations: {current_gamma_value}")

                # model.compile("adam", lr=LEARNING_RATE, loss_weights=LOSS_WEIGHTS)
                # model.train(iterations=20000, batch_size=BATCH_SIZE)
                # dde.optimizers.config.set_LBFGS_options(maxcor=50, ftol=1e-15, gtol=1e-10, maxiter=ITERATIONS_LBFGS, maxfun=50000, maxls=50)
                # model.compile("L-BFGS-B")
                # losshistory, train_state =model.train()

                gamma_values.append(current_gamma_value)
                iterations_list.append(total_iterations + iter_this_loop)

                # Inside your training loop, after training and updating gamma:
                predicted_h = model.predict(observed_xt)[:, 0]
                absolute_errors = np.abs(predicted_h - observed_h)
                mean_absolute_error = np.mean(absolute_errors)
                total_absolute_error = np.sum(absolute_errors)

                print(f"absolute_errors: {absolute_errors}")    
                print(f"mean absolute error: {mean_absolute_error}")
                print(f"total_absolute_error: {total_absolute_error}")

                # Store both errors in their respective lists
                mean_errors_over_iterations.append(mean_absolute_error)
                total_errors_over_iterations.append(total_absolute_error)

                # Print current lengths of the lists
                print(f"After iteration {total_iterations + iter_this_loop}:")

                # Adjust gamma based on total error trend
                #adjust_gamma_based_on_error(gamma_2_AC, total_absolute_error)


                
                # iteration += 1
                # # # Break condition for your loop
                # if iteration >= Max_iterations:
                #      break
                # # # Prepare for next loop
                total_iterations += iter_this_loop

            dde.optimizers.config.set_LBFGS_options(maxcor=50, ftol=1e-15, gtol=1e-10, maxiter=ITERATIONS_LBFGS, maxfun=50000, maxls=50)
            model.compile("L-BFGS-B")
            losshistory, train_state =model.train()
    #plot_error_and_gamma(gamma_values_history, total_errors_over_iterations)
    dde.saveplot(losshistory, train_state, issave=True, isplot=True)

    import pandas as pd
    import matplotlib.pyplot as plt

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

    return model



# plt.figure(figsize=(10, 6))
# plt.plot(iterations_list, gamma_values, marker='o', linestyle='-', color='blue')
# plt.xlabel('Iteration', fontsize=14)
# plt.ylabel('Gamma Value', fontsize=14)
# plt.title('Gamma Value Over Iterations', fontsize=16)
# plt.grid(True)
# plt.show()


# if len(gamma_values) > len(mean_errors_over_iterations):
#     gamma_values_corrected = gamma_values[1:]  # Skip initial gamma value
#     iterations_list_corrected = iterations_list[1:]  # Skip initial iteration number
# else:
#     gamma_values_corrected = gamma_values
#     iterations_list_corrected = iterations_list

# # Ensure corrected lists are used for plotting
# plt.figure(figsize=(10, 6))
# plt.plot(iterations_list_corrected, mean_errors_over_iterations, '-o', label='Mean Absolute Error', color='blue')
# plt.plot(iterations_list_corrected, total_errors_over_iterations, '-o', label='Total Absolute Error', color='red')
# plt.xlabel('Iteration', fontsize=14)
# plt.ylabel('Error', fontsize=14)
# plt.title('Error Over Iterations', fontsize=16)
# plt.legend()
# plt.grid(True)
# plt.show()

# plt.figure(figsize=(10, 6))
# if len(gamma_values_corrected) == len(mean_errors_over_iterations):
#     plt.plot(gamma_values_corrected, mean_errors_over_iterations, '-o', label='Mean Absolute Error', color='blue')
#     plt.plot(gamma_values_corrected, total_errors_over_iterations, '-o', label='Total Absolute Error', color='red')
#     plt.xlabel('Gamma Value', fontsize=14)
#     plt.ylabel('Error', fontsize=14)
#     plt.title('Error vs. Gamma Value', fontsize=16)
#     plt.legend()
#     plt.grid(True)
#     plt.show()
# else:
#     print("Mismatch in list sizes after correction:", len(gamma_values_corrected), len(mean_errors_over_iterations))





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
        #[bc_h, bc_h_deriv, initial_condition_h_AC],  # Include observe_h here
        [bc_h, bc_h_deriv, initial_condition_h_AC, observe_h_AC],  # Include observe_h here
        num_domain=20000,
        num_boundary=1600,
        num_initial=4096,
        anchors=observed_xt,  # Make sure observed_xt is used as anchors if necessary
        num_test=50000,
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
            model_new.train(iterations=30000, batch_size=BATCH_SIZE, disregard_previous_best=True, callbacks=[early_stopping])

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