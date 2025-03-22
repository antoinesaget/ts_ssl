import random
from pathlib import Path

import hydra
import matplotlib as mpl
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset

from ts_ssl.data.datamodule import (
    SSLGroupedTimeSeriesDataset,
    SupervisedGroupedTimeSeriesDataset,
)
from ts_ssl.utils.logger_manager import LoggerManager

# To use, run "python visualizer.py"
# Config defaults in ts_ssl/config/visualizer.yaml
# config.visualizer.function values:
#       - "plotbeforeandafter": plots one sample specified by config.visualizer.sample, before and after data augmentations are applied
#       - "plotsingleclass": plots one random sample of class specified by config.visualizer.classid (0-19)
#       - "plotmulticlass": plots one random sample per class in dataset (20 total)
#
# Plot display/saving options:
#       - config.visualizer.save_plot: when true, plots are saved to the output directory instead of displaying
#                                      when false (default), plots are displayed in interactive windows


@hydra.main(config_path="../config", config_name="config", version_base="1.3")
def visualize(config):
    logger = LoggerManager(
        output_dir=Path(config.output_dir),
        loggers=config.logging.enabled_loggers,
        log_file=Path(config.output_dir) / "visualize.log",
    )

    func = config.visualizer.function
    save_plot = (
        config.visualizer.save_plot
        if hasattr(config.visualizer, "save_plot")
        else False
    )

    # Determine function to use; initialize dataset appropriately (SSL for beforeandafter, Supervised otherwise)
    if func == "plotbeforeandafter":
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
        plotbeforeandafter(
            dataset=dataset,
            sample=config.visualizer.sample,
            overlay=config.visualizer.overlay,
            augmentation=config.augmentations.name,
            bands=config.visualizer.bands,
            save_plot=save_plot,
            output_dir=Path(config.output_dir),
            logger=logger,
        )
    elif func == "plotsingleclass":
        logger.info("Loading dataset")
        dataset = SupervisedGroupedTimeSeriesDataset(
            data=config.dataset.validation_data,
            percentiles=config.dataset.percentiles,
            logger=logger,
            normalize_data=config.dataset.normalize,
            dataset_type=config.dataset.dataset_type,
        )
        plotsingleclassexamples(
            dataset=dataset,
            classid=config.visualizer.classid,
            save_plot=save_plot,
            output_dir=Path(config.output_dir),
            logger=logger,
        )
    elif func == "plotmulticlass":
        logger.info("Loading dataset")
        dataset = SupervisedGroupedTimeSeriesDataset(
            data=config.dataset.validation_data,
            percentiles=config.dataset.percentiles,
            logger=logger,
            normalize_data=config.dataset.normalize,
            dataset_type=config.dataset.dataset_type,
        )
        plotmulticlassexamples(
            dataset=dataset,
            save_plot=save_plot,
            output_dir=Path(config.output_dir),
            logger=logger,
        )


def plotmulticlassexamples(
    dataset: Dataset, save_plot=False, output_dir=None, logger=None
) -> None:
    """
    Plots example from each class of a dataset.

    Parameters:
        dataset: dataset to select data from with feature shape (100, 60, 12)
        save_plot: if True, save plot to file instead of displaying
        output_dir: directory to save plot (only used if save_plot is True)
        logger: LoggerManager instance for logging

    Returns:
        None
    """
    # Get list of sample from each class, create list [0...59] for plotting
    samples = _getmulticlassexamples(dataset, logger=logger)
    temporal_dim = list(range(60))

    # Create 5x4 subplot figure and init row and col indexes
    fig, axes = plt.subplots(
        nrows=5, ncols=4, sharex=True, sharey=True, layout="constrained"
    )
    fig.set_size_inches(10, 15)
    row, col = (0, 0)

    # For each class sample in list of samples
    for sample in samples:
        x = sample[0][0][0]  # (2, 1, 100)
        y = sample[1]

        # Allign spectral band measurements appropriately
        spectral_bands = [list(indices) for indices in zip(*x)]

        # Plot data on appropriate graph
        axes[row, col].set_title(f"Class {int(y)}")
        for band in range(len(spectral_bands)):
            axes[row, col].plot(
                temporal_dim, spectral_bands[band], label=f"Band {band + 1}"
            )

        # Update rows, columns index
        if col < 3:
            col += 1
        elif col == 3:
            col = 0
            row += 1
        else:
            raise IndexError("Row/column index out of range")

    if save_plot and output_dir:
        plot_path = output_dir / "multiclass_examples.png"
        plt.savefig(plot_path)
        if logger:
            logger.info(f"Plot saved to {plot_path}")
    else:
        plt.show()
    return


def _getmulticlassexamples(dataset: Dataset, n_classes: int = 20, logger=None):
    """Helper function to obtain a sample from each class of a dataset. Returns list of same length as number of classes in the dataset"""
    if logger:
        logger.info("Selecting example from each class")
    samples = {}

    sampler = DataLoader(dataset=dataset, shuffle=True)

    for sample in sampler:
        _, y = sample
        classid = int(y)
        if classid not in samples.keys():
            samples[classid] = sample
        if len(samples) == n_classes:
            break

    # Sort samples by class id
    sorted_samples = sorted(samples.items())

    return [sample for _, sample in sorted_samples]


def plotsingleclassexamples(
    dataset: Dataset, classid: int, save_plot=False, output_dir=None, logger=None
) -> None:
    """
    Plots 10 examples from one specified class of a dataset.

    Parameters:
        dataset: dataset to select data from with feature shape (100, 60, 12)
        classid: integer of targeted class
        save_plot: if True, save plot to file instead of displaying
        output_dir: directory to save plot (only used if save_plot is True)
        logger: LoggerManager instance for logging

    Returns:
        None
    """
    # Get samples of specified class, get the timeseries data from each
    samples = [
        sample[0][0][0]
        for sample in _getsingleclassexamples(dataset, classid, logger=logger)
    ]
    temporal_dim = list(range(60))

    # Create 5x4 subplot figure and init row and col indexes
    fig, axes = plt.subplots(
        nrows=2, ncols=5, sharex=True, sharey=True, layout="constrained"
    )
    fig.set_size_inches(12, 5)
    fig.suptitle(f"10 samples of class {classid}")
    row, col = (0, 0)

    # For each class sample in list of samples
    for sample in samples:
        # Access x and y features
        x = sample.tolist()

        # Allign spectral band measurements appropriately
        spectral_bands = [list(indices) for indices in zip(*x)]

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

    if save_plot and output_dir:
        plot_path = output_dir / f"class_{classid}_examples.png"
        plt.savefig(plot_path)
        if logger:
            logger.info(f"Plot saved to {plot_path}")
    else:
        plt.show()
    return


def _getsingleclassexamples(dataset: Dataset, classid: int, logger=None):
    """Helper function to return ten samples of specified class, each with feature shape dependant on dataset type."""
    if logger:
        logger.info("Selecting examples from specified class")
    # Init list to be returned
    samples = []

    # Create sampler to shuffle mmap data
    sampler = DataLoader(dataset=dataset, shuffle=True)

    # Iterate until 10 samples of same class are found
    for sample in sampler:
        _, y = sample
        if y == classid:
            samples.append(sample)
        if len(samples) == 10:
            break

    return samples


# masking, resampling, resizing, combination, jittering
def plotbeforeandafter(
    dataset: SSLGroupedTimeSeriesDataset,
    sample: int = -1,
    augmentation: str = "augmentation",
    overlay: bool = True,
    bands: list[int] | None = None,
    save_plot=False,
    output_dir=None,
    logger=None,
) -> None:
    """
    Displays a two plot visualization of a sample from an SSLGroupedTimeSeriesDataset, one before augmentations are applied, the other after.
    Augmentation determined by "augmentations" config setting.
    Works well with "dataset.normalize" config setting set to "False"

    Parameters:
        dataset: pytorch Dataset with feature shape (100, 60, 12)
        sample: index of sample within dataset in bounds [0, 99]
        augmentation: name of applied augmentation
        overlay: if true, before/after data displayed on same graph; if false, before/after data displayed on separate graphs
        bands: list of spectral bands to be displayed; defaults to all bands
        save_plot: if True, save plot to file instead of displaying
        output_dir: directory to save plot (only used if save_plot is True)
        logger: LoggerManager instance for logging

    Returns:
        None
    """
    # Initialize bands to be displayed
    if not bands:
        bands = [x for x in list(range(12))]
    if sample == -1:
        sample = random.randint(0, len(dataset))

    # Sample of one time series (len=60)
    sample_before = dataset.get_raw_item(sample)[0][0].tolist()
    sample_after = dataset[sample][0][0].tolist()

    # List [0,..., 59], time points
    temporal_dim = list(range(len(sample_before)))

    # Sort values of same spectral bands into 2d list shape (12, 60); Method found on GeeksforGeeks "Group Elements at Same Indices in a Multi-List"
    spectral_bands_before = [list(indices) for indices in zip(*sample_before)]
    spectral_bands_after = [list(indices) for indices in zip(*sample_after)]

    # Two stacked plots
    if not overlay:
        # Define matplotlib subplots; always shares x axes, shares y axes if augmented data is not normalized
        fig, (ax0, ax1) = plt.subplots(
            2, sharex=True, sharey=not dataset.normalize_data, layout="constrained"
        )
        ax0.set_title(f"Sample before {augmentation}")
        ax1.set_title(f"Sample after {augmentation}")

        # Add each band to the plot
        for band in bands:
            ax0.plot(temporal_dim, spectral_bands_before[band])
            ax1.plot(temporal_dim, spectral_bands_after[band])

        # Set legend labels
        fig.legend([f"Band {band + 1}" for band in bands], loc="outside left upper")

    # Overlayed plots
    else:
        if dataset.normalize_data:
            error_msg = "Overlay and dataset normalization are mutually exclusive"
            if logger:
                logger.error(error_msg)
            raise ValueError(error_msg)

        # Define matplotlib plot
        fig, axe = plt.subplots(layout="constrained")
        axe.set_title(f"Sample before and after {augmentation}")

        # Add each band to the plot
        for band in bands:
            axe.plot(
                temporal_dim,
                spectral_bands_before[band],
                color=mpl.color_sequences["tab20"][band],
                label=f"Band {band + 1}",
            )
            axe.plot(
                temporal_dim,
                spectral_bands_after[band],
                color=mpl.color_sequences["tab20"][band],
                linestyle="dotted",
            )

        # fig.legend([])
        color_legend = fig.legend(loc="outside left upper")
        fig.add_artist(color_legend)
        fig.legend(
            handles=[
                axe.plot([0], linestyle="solid", color="black", label="Before")[0],
                axe.plot([0], linestyle="dotted", color="black", label="After")[0],
            ],
            loc="outside left lower",
        )

    if save_plot and output_dir:
        sample_str = f"sample_{sample}"
        plot_path = output_dir / f"beforeafter_{augmentation}_{sample_str}.png"
        plt.savefig(plot_path)
        if logger:
            logger.info(f"Plot saved to {plot_path}")
    else:
        plt.show()
    return


if __name__ == "__main__":
    visualize()
