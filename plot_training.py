import json
import matplotlib.pyplot as plt
import torch
import numpy as np
import os


def load_data(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            data.append(json.loads(line))
    return data


def plot_parameter_old(data, parameter, label, color, skip_iter=1000):
    iterations = [entry['iter'] for entry in data]
    values = [entry[parameter] for entry in data]
    epochs = [entry['ep'] for entry in data]
    
    fig, ax1 = plt.subplots(figsize=(18, 5))

     # Plotting the main parameter on the primary x-axis
    ax1.plot(iterations, values, label=label, color=color)
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel(label)
    ax1.set_title(f'{label} over Iterations')

    # Adding vertical lines for each epoch and text labels
    for i, (iteration, epoch) in enumerate(zip(iterations, epochs)):
        if i % skip_iter == 0:  # Add a line and label for every 100 iterations
            ax1.axvline(x=iteration, color='grey', linestyle='--', linewidth=0.5)
            ax1.text(iteration, max(values) * 0.95, f'Ep {epoch}', rotation=90, verticalalignment='center')

    # Setting x-ticks and rotating tick labels
    ax1.set_xticks([iterations[i] for i in range(0, len(iterations), skip_iter)])
    ax1.set_xticklabels([iterations[i] for i in range(0, len(iterations), skip_iter)], rotation=45)

    fig.savefig(path_out+parameter+'.png', bbox_inches='tight')
    plt.close()



def plot_parameter(data, parameter, label, color, path_out, skip_iter=2000):
    iterations = [entry['iter'] for entry in data]
    values = [entry[parameter] for entry in data]
    epochs = [entry['ep'] for entry in data]
    
    fig, ax1 = plt.subplots(figsize=(18, 5))

    # Plotting the main parameter on the primary x-axis
    ax1.plot(iterations, values, label=label, color=color)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel(label)
    ax1.set_title(f'{label} over Epochs')

    # Setting x-ticks at intervals of skip_iter and replacing them with epoch labels
    tick_positions = [iterations[i] for i in range(0, len(iterations), skip_iter)]
    tick_labels = [epochs[i] for i in range(0, len(epochs), skip_iter)]
    
    ax1.set_xticks(tick_positions)
    ax1.set_xticklabels(tick_labels, rotation=45)

    fig.savefig(f'{path_out}{parameter}.png', bbox_inches='tight')
    plt.close()

def plot_parameter_avg(data, parameter, label, color, path_out):
    # Group data by epochs and compute the average value for each epoch
    epoch_dict = {}
    for entry in data:
        epoch = entry['ep']
        if epoch not in epoch_dict:
            epoch_dict[epoch] = []
        epoch_dict[epoch].append(entry[parameter])

    # Compute the average value for each epoch
    epochs = sorted(epoch_dict.keys())
    avg_values = [np.mean(epoch_dict[epoch]) for epoch in epochs]

    # Plotting the averaged values
    fig, ax1 = plt.subplots(figsize=(18, 5))
    ax1.plot(epochs, avg_values, label=label, color=color)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel(label)
    ax1.set_title(f'Average {label} over Epochs')

    # Save the figure
    fig.savefig(f'{path_out}{parameter}_avg_per_epoch.png', bbox_inches='tight')
    plt.close()



def plot_parameter_multiple_datasets(datasets, parameter, labels, colors, path_out):
    """
    Plots the averaged values of a single parameter from multiple datasets over epochs on a single plot.

    :param datasets: List of datasets, each containing dictionaries with 'ep' (epoch) and parameter values.
    :param parameter: The parameter to plot from each dataset.
    :param labels: List of labels corresponding to each dataset.
    :param colors: List of colors for each dataset's plot.
    :param path_out: Path to save the plot.
    """
    fig, ax1 = plt.subplots(figsize=(18, 5))
    
    for data, label, color in zip(datasets, labels, colors):
        # Create a dictionary to store average values for the parameter by epoch
        avg_values = []
        epochs = sorted(set(entry['ep'] for entry in data))
        
        for epoch in epochs:
            # Filter data for the current epoch
            epoch_data = [entry for entry in data if entry['ep'] == epoch]
            
            # Compute the average value for the parameter in the current epoch
            avg_value = np.mean([entry[parameter] for entry in epoch_data])
            avg_values.append(avg_value)
        
        # Plotting the parameter
        ax1.plot(epochs, avg_values, label=label, color=color)
    
    # Setting labels and title
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel(parameter)
    ax1.set_title(f'Average {parameter} over Epochs')

    # Enable grid
    ax1.grid(True)
    
    # Add legend
    ax1.legend()

    # Save the figure
    fig.savefig(f'{path_out}{parameter}_multiple_datasets_avg_per_epoch.png', bbox_inches='tight')
    plt.close()


path_out = '/home/Daniele/fig/' #dcv_ir108_128x128_k9_30k_grey_5th-95th/'


# Check if the directory exists
if not os.path.exists(path_out):
    # Create the directory if it doesn't exist
    os.makedirs(path_out) 

# Initialize the list to store datasets
datasets = []

labels = ['min-max', '1th-99th', '5th-95th']
colors = ['red','blue','green']

#for each case open data 

for label in labels:
    # Load the data from the JSON file
    file_path = f'/home/Daniele/codes/vissl/runs/dcv2_ir108_128x128_k9_germany_30kcrops_grey_{label}/checkpoints/'
    data = []

    #collect data from jason output file
    data = load_data(file_path+'stdout.json')

    # Add the loaded data to the datasets list
    if data:
        datasets.append(data)

plot_parameter_multiple_datasets(datasets, 'loss', labels, colors, path_out)

# Plot learning rate
#plot_parameter(data, 'lr', 'Learning Rate', 'blue', path_out)

# plot loss:
#plot_parameter_avg(data, 'loss', 'Loss', 'red', path_out)


# filename = 'model_final_checkpoint_phase799.torch'


# data = torch.load(file_path+filename)
# #data = np.load(file_path+filename)

# print(data)


