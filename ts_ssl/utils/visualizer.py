import logging

import matplotlib.pyplot as plt
import matplotlib as mpl
import torch
import hydra
from torch.utils.data import Dataset
from ts_ssl.data.datamodule import SSLGroupedTimeSeriesDataset

# To use, run "python visualizer.py"
# Config defaults in utils/visualizer_config/config.yaml
# config.visualizer.function values:
#       - "plotbeforeandafter": plots one sample specified by config.visualizer.sample, before and after data augmentations are applied
#       - "plotsingleclass": plots one random sample of class specified by config.visualizer.classid (0-19)
#       - "plotmulticlass": plots one random sample per class in dataset (20 total)

@hydra.main(config_path="visualizer_config", config_name="config", version_base="1.3")
def visualize(config):
    logger = logging.getLogger(__name__)
    # Initialize datasets and dataloaders
    logger.info("Loading dataset")
    dataset = SSLGroupedTimeSeriesDataset(
        data=config.dataset.ssl_data,
        n_samples_per_group=config.training.n_samples_per_group,
        percentiles=config.dataset.percentiles,
        config=config.augmentations,
        logger=logger,
        normalize_data=config.dataset.normalize,
        dataset_type=config.dataset.dataset_type,
    )

    func = config.visualizer.function
    if func == "plotbeforeandafter":
        plotbeforeandafter(dataset=dataset, sample=config.visualizer.sample, overlay=config.visualizer.overlay, 
                           augmentation=config.augmentations.name, bands=config.visualizer.bands)
    elif func == "plotsingleclass":
        plotsingleclassexamples(dataset=dataset, classid=config.visualizer.seed)
    elif func == "plotmulticlass":
        plotmulticlassexamples(dataset=dataset, augmentation=config.augmentations, shuffle_seed=config.visualizer.seed)

def plotdata(dataset: Dataset) -> None:
    '''
    Displays a plot representation of a pytorch Dataset
    
    Parameters:
        dataset: pytorch Dataset with feature shape (100, 60, 12)

    Returns:
        None
    '''
    # sample of one time series (len=60)
    sample = dataset[0][0][0].tolist()
    # List [0,..., 59], time points
    temporal_dim = list(range(len(sample)))
    # Sort values of same spectral bands into 2d list shape (12, 60); Method found on GeeksforGeeks "Group Elements at Same Indices in a Multi-List"
    spectral_bands=[list(indices) for indices in zip(*sample)]
    # Add each band to the plot
    for idx in range(12):
        plt.plot(temporal_dim, spectral_bands[idx], label=f"Band {idx}")
    plt.legend()
    plt.show()
    return

def plotdata(datasets: list[Dataset]) -> None:
    '''
    Displays a multi-plot representation of a pytorch Dataset
    
    Parameters:
        dataset: pytorch Dataset with feature shape (100, 60, 12)

    Returns:
        None
    '''
    figs, axes = plt.subplots(len(datasets))
    for dataset_idx in range(len(datasets)):
        # Get individual dataset from list argument 
        dataset = datasets[dataset_idx]
        # Sample of one time series (len=60)
        sample = dataset[0]["x"][0].tolist() # HuggingFace dataset
        # List [0,..., 59], time points
        temporal_dim = list(range(len(sample)))
        # Sort values of same spectral bands into 2d list shape (12, 60); Method found on GeeksforGeeks "Group Elements at Same Indices in a Multi-List"
        spectral_bands=[list(indices) for indices in zip(*sample)]
        # Add each band to the plot
        for idx in range(12):
            axes[dataset_idx].plot(temporal_dim, spectral_bands[idx], label=f"Band {idx}")
    plt.legend()
    plt.show()
    return

def plotmulticlassexamples(dataset: Dataset, shuffle_seed=None) -> None:
    '''
    Plots example from each class of a dataset.

    Parameters:
        dataset: dataset to select data from with feature shape (100, 60, 12)
        shuffle_seed: integer seed to shuffle set for selection

    Returns:
        None
    '''
    # Get list of sample from each class, create list [0...59] for plotting
    samples = _getmulticlassexamples(dataset, shuffle_seed=shuffle_seed)
    temporal_dim = list(range(60))

    # Create 5x4 subplot figure and init row and col indexes
    fig, axes = plt.subplots(nrows=5, ncols=4, sharex=True, sharey=True, layout="constrained")
    fig.set_size_inches(10,15)
    row, col = (0,0)

    # For each class sample in list of samples
    for sample in samples:
        # Access x and y features
        x = sample["x"][0].tolist()
        y = sample["y"].tolist()

        # Allign spectral band measurements appropriately
        spectral_bands=[list(indices) for indices in zip(*x)]

        # Plot data on appropriate graph
        axes[row, col].set_title(f"Class {int(y)}")
        for band in range(len(spectral_bands)):
            axes[row, col].plot(temporal_dim, spectral_bands[band], label=f"Band {band+1}")
        
        # Update rows, columns index
        if col < 3:
            col += 1
        elif col == 3:
            col = 0
            row += 1
        else:
            raise IndexError("Row/column index out of range")
        
    plt.show()
    return

def _getmulticlassexamples(dataset: Dataset, shuffle_seed = None):
    '''Helper function to obtain a sample from each class of a dataset. Returns list of same length as number of classes in the dataset'''
    logging.getLogger(__name__).info("Selecting example from each class")
    samples = []
    for classnum in range(20):
        filtered_set = dataset.data.filter(lambda x: x["y"] == torch.tensor(classnum))
        samples.append(filtered_set.shuffle(seed=shuffle_seed)[0])
    return samples

def plotsingleclassexamples(dataset: Dataset, classid: int, shuffle_seed: int = None) -> None:
    '''
    Plots 10 examples from one specified class of a dataset.

    Parameters:
        dataset: dataset to select data from with feature shape (100, 60, 12)
        classid: integer of targeted class
        shuffle_seed: integer seed to shuffle set for selection

    Returns:
        None
    '''
    # Get sample of specified class, get the first 20 timeseries from the features
    samples = _getsingleclassexamples(dataset, classid, shuffle_seed=shuffle_seed)["x"][0:10]
    temporal_dim = list(range(60))

    # Create 5x4 subplot figure and init row and col indexes
    fig, axes = plt.subplots(nrows=2, ncols=5, sharex=True, sharey=True, layout="constrained")
    fig.set_size_inches(12,5)
    fig.suptitle(f"10 samples of class {classid}")
    row, col = (0,0)

    # For each class sample in list of samples
    for sample in samples:
        # Access x and y features
        x = sample.tolist()

        # Allign spectral band measurements appropriately
        spectral_bands=[list(indices) for indices in zip(*x)]

        # Plot data on appropriate graph
        for band in range(len(spectral_bands)):
            axes[row, col].plot(temporal_dim, spectral_bands[band])
        
        # Update rows, columns index
        if col < 4:
            col += 1
        elif col == 4:
            col = 0
            row += 1
        else:
            raise IndexError("Row/column index out of range")
        
    plt.show()
    return

def _getsingleclassexamples(dataset: Dataset, classid: int, shuffle_seed: int = None):
    '''Helper function to return one sample of specified class, with feature shape (100, 60, 12)'''
    logging.getLogger(__name__).info("Selecting example from each class")
    # Filter set with condition input label == classid
    filtered_set = dataset.data.filter(lambda x: x["y"] == torch.tensor(classid))
    # Shuffle set and choose first sample (feature shape (100,60,12))
    samples = filtered_set.shuffle(shuffle_seed)[0]
    return samples

# masking, resampling, resizing, combination, jittering
def plotbeforeandafter(dataset: Dataset, sample: int = 0, augmentation: str = "augmentation", overlay: bool = True, 
                       bands: list[int] = None) -> None:
    '''
    Displays a two plot visualization of a sample from an SSLGroupedTimeSeriesDataset, one before augmentations are applied, the other after. 
    Augmentation determined by "augmentations" config setting.
    Works well with "dataset.normalize" config setting set to "False"
    
    Parameters:
        dataset: pytorch Dataset with feature shape (100, 60, 12)
        sample: index of sample within dataset in bounds [0, 99]
        augmentation: name of applied augmentation
        overlay: if true, before/after data displayed on same graph; if false, before/after data displayed on separate graphs
        bands: list of spectral bands to be displayed; defaults to all bands

    Returns:
        None
    '''
    # Initialize bands to be displayed
    if not bands:
        bands = [x for x in list(range(12))]

    # Sample of one time series (len=60)
    sample_before = dataset.data[sample]["x"][0].tolist()   # HFDataset format
    sample_after = dataset[sample][0][0].tolist()   # Pytorch Dataset  

    # List [0,..., 59], time points
    temporal_dim = list(range(len(sample_before)))

    # Sort values of same spectral bands into 2d list shape (12, 60); Method found on GeeksforGeeks "Group Elements at Same Indices in a Multi-List"
    spectral_bands_before=[list(indices) for indices in zip(*sample_before)]
    spectral_bands_after=[list(indices) for indices in zip(*sample_after)]

    # Two stacked plots
    if not overlay:
        # Define matplotlib subplots; always shares x axes, shares y axes if augmented data is not normalized 
        fig, (ax0, ax1) = plt.subplots(2, sharex=True, sharey=not dataset.normalize_data, layout="constrained")
        ax0.set_title(f"Sample before {augmentation}")
        ax1.set_title(f"Sample after {augmentation}")

        # Add each band to the plot
        for band in bands:
            ax0.plot(temporal_dim, spectral_bands_before[band])
            ax1.plot(temporal_dim, spectral_bands_after[band])

        # Set legend labels
        fig.legend([f"Band {band+1}" for band in bands], loc="outside left upper")   

    # Overlayed plots
    else:
        if dataset.normalize_data:
            raise ValueError("Overlay and dataset normalization are mutually exclusive")
        
        # Define matplotlib plot
        fig, axe = plt.subplots(layout="constrained")
        axe.set_title(f"Sample before and after {augmentation}")

        # Add each band to the plot
        for band in bands:
            axe.plot(temporal_dim, spectral_bands_before[band], color=mpl.color_sequences["tab20"][band], label=f"Band {band+1}")
            axe.plot(temporal_dim, spectral_bands_after[band],  color=mpl.color_sequences["tab20"][band], linestyle="dotted")
        
        #fig.legend([])
        color_legend = fig.legend(loc="outside left upper")
        fig.add_artist(color_legend)
        fig.legend(handles=[axe.plot([0], linestyle="solid", color="black", label="Before")[0], 
                             axe.plot([0], linestyle="dotted", color="black", label="After")[0]], 
                             loc="outside left lower")

    plt.show()
    return

if __name__ == "__main__":
    visualize()

