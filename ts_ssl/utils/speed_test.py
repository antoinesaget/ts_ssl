import hydra
import mmap_ninja
import torch
import logging
from tqdm import tqdm

import cProfile, pstats, io
from pstats import StatsProfile
from cProfile import Profile
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from ts_ssl.data.datamodule import SSLGroupedTimeSeriesDataset


@hydra.main(config_path="../config", config_name="config", version_base="1.3")
def main(config):
    logger = logging.getLogger(__name__)

    function = config.dataloaderutil.function

    if function == "runarrowmmap":
        run_arrow_mmap(config, logger)
    elif function == "runaugmentations":
        run_augmentations(config, logger)
    elif function == "runnumworkers":
        run_num_workers(config, logger)
    elif function == "runbatchsize":
        run_batch_size(config, logger)
    else:
        ValueError("Config value dataloaderutil.function must be runarrowmmap|runaugmentations|runnumworkers")

    logger.info(f"Log file may be found at {config.output_dir}")


def run_arrow_mmap(config, logger = None) -> None:
    '''
    Times and compares iteration through a dataset of parquet arrow format and one of numpy memmap format
    '''
    # Get logger if not provided
    if logger is None:
        logger = logging.getLogger(__name__)

    # Access configs for each
    arrow_cfg = config.dataloaderutil.arrow
    mmap_cfg = config.dataloaderutil.mmap

    # Create dataset objects
    dataset_arrow = SSLGroupedTimeSeriesDataset(
        data=arrow_cfg.ssl_data,
        n_samples_per_group=config.training.n_samples_per_group,
        percentiles=arrow_cfg.percentiles,
        config=config.augmentations,
        normalize_data=arrow_cfg.normalize,
        dataset_type=arrow_cfg.dataset_type,
        )
    dataset_mmap = SSLGroupedTimeSeriesDataset(
        data=mmap_cfg.ssl_data,
        n_samples_per_group=config.training.n_samples_per_group,
        percentiles=mmap_cfg.percentiles,
        config=config.augmentations,
        normalize_data=mmap_cfg.normalize,
        dataset_type=mmap_cfg.dataset_type,
        )
    
    # Create dataloader objects
    dataloader_arrow = DataLoader(dataset=dataset_arrow, batch_size=config.training.batch_size, num_workers=config.num_workers)
    dataloader_mmap = DataLoader(dataset=dataset_mmap, batch_size=config.training.batch_size, num_workers=config.num_workers)

    # Time runs
    logger.info("Timing arrow (huggingface) type dataset")
    data_arrow = time_runs(dataloader_arrow, config.dataloaderutil.iterations)
    logger.info("Timing mmap type dataset")
    data_mmap = time_runs(dataloader_mmap, config.dataloaderutil.iterations)

    # Compare data
    compare(config, data_arrow, data_mmap)

    return


def run_augmentations(config, logger = None) -> None:
    '''
    Times and compares iteration through datasets with augmentations specified in the config
    '''
    if not (0 < len(config.dataloaderutil.augmentations) < 4):    
        raise ValueError("Config values dataloaderutil.augmentations must have length in [1, 3]")
    
    if not logger:
        logger = logging.getLogger(__name__)

    augmentation_map = {
        "combination":config.dataloaderutil.combination,
        "jittering":config.dataloaderutil.jittering,
        "masking":config.dataloaderutil.masking,
        "resampling":config.dataloaderutil.resampling,
        "resizing":config.dataloaderutil.resizing,
        }

    dataset_cfg = config.dataloaderutil.mmap if config.dataloaderutil.dataset_type == "mmap" else config.dataloaderutil.arrow

    data = []
    for augmentation in config.dataloaderutil.augmentations:
        if augmentation not in augmentation_map.keys():
            raise ValueError(f"Augmentation \"{augmentation}\" not valid; valid: resizing|resampling|combination|jittering|masking")
        
        augmentation = augmentation_map[augmentation]

        logger.info(f"Loading dataset with augmentation \"{augmentation.name}\"...")
        dataset = SSLGroupedTimeSeriesDataset(
            data=dataset_cfg.ssl_data,
            n_samples_per_group=config.training.n_samples_per_group,
            percentiles=dataset_cfg.percentiles,
            config=augmentation,
            normalize_data=dataset_cfg.normalize,
            dataset_type=dataset_cfg.dataset_type,
            )
        dataloader = DataLoader(dataset=dataset, batch_size=config.training.batch_size, num_workers=config.num_workers)
        logger.info(f"Timing dataset with augmentation \"{augmentation.name}\"...")
        profile_data = time_runs(loader=dataloader, num_times=config.dataloaderutil.iterations)
        data.append(profile_data)
    
    compare(config, data)

    return


def run_num_workers(config, logger = None) -> None:
    '''
    Times and compares iteration through a dataset with num_workers of dataloaders specified in the config
    '''
    if not (0 < len(config.dataloaderutil.num_workers) < 4):    
        raise ValueError("Config values dataloaderutil.num_workers must have length in [1, 3]")

    if not logger:
        logger = logging.getLogger(__name__)
    
    dataset_cfg = config.dataloaderutil.mmap if config.dataloaderutil.dataset_type == "mmap" else config.dataloaderutil.arrow
    dataset = SSLGroupedTimeSeriesDataset(
            data=dataset_cfg.ssl_data,
            n_samples_per_group=config.training.n_samples_per_group,
            percentiles=dataset_cfg.percentiles,
            config=config.augmentations,
            normalize_data=dataset_cfg.normalize,
            dataset_type=dataset_cfg.dataset_type,
            )
    
    data = []
    for num_workers in config.dataloaderutil.num_workers:
        if num_workers < 1:
            raise ValueError("Config values dataloaderutil.num_workers must contain positive integer values")
        
        dataloader = DataLoader(dataset=dataset, batch_size=config.training.batch_size, num_workers=num_workers)
        logger.info(f"Timing dataset with {num_workers} loader workers...")
        profile_data = time_runs(loader=dataloader, num_times=config.dataloaderutil.iterations)
        data.append(profile_data)
    
    compare(config, data)
    return


def run_batch_size(config, logger = None) -> None:
    '''
    Times and compares iteration through a dataset with batch_size of dataloaders specified in the config
    '''
    if not (0 < len(config.dataloaderutil.batch_sizes) < 4):    
        raise ValueError("Config values dataloaderutil.batch_sizes must have length in [1, 3]")

    if not logger:
        logger = logging.getLogger(__name__)
    
    dataset_cfg = config.dataloaderutil.mmap if config.dataloaderutil.dataset_type == "mmap" else config.dataloaderutil.arrow
    dataset = SSLGroupedTimeSeriesDataset(
            data=dataset_cfg.ssl_data,
            n_samples_per_group=config.training.n_samples_per_group,
            percentiles=dataset_cfg.percentiles,
            config=config.augmentations,
            normalize_data=dataset_cfg.normalize,
            dataset_type=dataset_cfg.dataset_type,
            )
    
    data = []
    for batch_size in config.dataloaderutil.batch_sizes:
        if batch_size < 1:
            raise ValueError("Config values dataloaderutil.num_workers must contain positive integer values")
        
        dataloader = DataLoader(dataset=dataset, batch_size=batch_size, num_workers=config.num_workers)

        logger.info(f"Timing dataset with loader batch size of {batch_size}...")
        profile_data = time_runs(loader=dataloader, num_times=config.dataloaderutil.iterations)

        data.append(profile_data)

    compare(config, data)
    return


def time_runs(loader: DataLoader, num_times: int, print: bool = False, logger = None):
    '''
    Iterates through an entire dataset using DataLoader <num_times> number of times. Timed using cProfile; returns pstats data from cProfile

    Arguments:
        loader: pytorch DataLoader
        num_times: integer number of iterations

    Returns:
        Tuple = (dataset type, num_times, StatsProfile of run)
    '''
    # Get logger and IO to store profile stats
    if not logger:
        logger = logging.getLogger(__name__)
    stats_stream = io.StringIO()

    # Using a profiler, iterate through the dataset <num_times> times
    with tqdm(total=(num_times*len(loader))) as pbar:    #   <- undecided on usage
        with cProfile.Profile() as pr:
            for iteration in range(num_times):
                for item in loader:
                    len(item)
                    pbar.update(1)

    # Obtain the profiler stats, format, and print to IO 
    profile_stats = pstats.Stats(pr, stream=stats_stream).sort_stats("cumulative")
    profile_stats.print_stats(15)

    # Use IO to print stats to logger
    if print:
        logger.info(f"Iterated {num_times} time(s) through dataset of size {len(loader.dataset)} with a batch size of {loader.batch_size} and {loader.num_workers} worker(s): \n\n{stats_stream.getvalue()}")
    
    return loader, pr


def compare(config, *profiles: tuple[DataLoader, Profile]):
    '''
    Compares up to three runs from time_runs(); prints information to log in readable form.

    Arguments:
        *profiles: array of tuples in format (pytorch DataLoader, cProfile Profile object)
    
    Returns:
        None: Prints data to log
    '''

    # Check if passed a list instead of multiple tuples
    if len(profiles) == 1 and isinstance(profiles[0], list):
        profiles = profiles[0]

    if len(profiles) > 3:
        raise ValueError("More than three profiles not supported")

    # Initialize reused values
    logger = logging.getLogger(__name__)
    stream = io.StringIO()
    results = []
    col_width = 55

    iterations = config.dataloaderutil.iterations

    # For each tuple in *profiles argument
    for idx, group in enumerate(profiles):
        dataset = group[0].dataset
        # Break apart tuple data
        dataset_type = dataset.dataset_type if hasattr(dataset, "dataset_type") else "N/A"
        dataset_class = str(type(dataset))
        dataset_size = len(dataset)
        dataset_dtype = dataset.get_dtype()
        dataset_augmentations = dataset.get_augmentations()
        dataloader_num_workers = group[0].num_workers
        dataloader_batch_size = group[0].batch_size
        profile = group[1]

        # Create pstats objects and print stats to io stream
        stats = pstats.Stats(profile, stream=stream)
        stream.write(F"Profile {idx+1}: {dataset_type}, {dataset_class}, {dataset_augmentations}\n")
        stats.sort_stats("cumulative").print_stats(20)
        stats_profile = stats.get_stats_profile()

        # Find total time and average time per complete iteration
        total_time = stats_profile.total_tt
        time_per_iter = total_time/iterations

        # Sort function profiles from pstats StatsProfile by cumulative time, get first 5
        top_five_funcs = sorted(stats_profile.func_profiles.items(), key=lambda x: x[1].cumtime, reverse=True)[:7]

        # Create string presentation of each function; lists function name, file, line, and cumulative time
        top_funcs_info = [f"{fp[0][:18]+"..." if len(fp[0]) > 21 else fp[0]}".ljust(22) + f"{"..."+fp[1].file_name[-(18-len(str(fp[1].line_number))):] if len(fp[1].file_name+":"+str(fp[1].line_number)) > 21 else fp[1].file_name}:{fp[1].line_number} - {fp[1].cumtime}s".rjust(33) for fp in top_five_funcs]

        results.append((dataset_class, dataset_type, dataset_size, dataset_dtype, 
                        dataset_augmentations, dataloader_num_workers, dataloader_batch_size,
                        total_time, time_per_iter, top_funcs_info))

    # Separator string between rows
    # separator = " | ".join(["-" * col_width] * len(results))

    # Write header
    headers = [f"Profile {i+1}" for i in range(len(results))]
    stream.write(" | ".join(h.center(col_width) for h in headers) + "\n")

    # Config data
    stream.write(" | ".join([" Config ".center(col_width, "-")] * len(results)) + "\n")
    
    num_workers = [f"num_workers:" + f"{result[5]}".rjust(col_width-12) for result in results]
    stream.write(" | ".join(num_workers) + "\n")

    batch_size = [f"batch_size:" + f"{result[6]}".rjust(col_width-11) for result in results]
    stream.write(" | ".join(batch_size) + "\n")

    augmentations = [f"augmentations:" + f"{result[4]}".rjust(col_width-14) for result in results]
    stream.write(" | ".join(augmentations) + "\n")

    # Dataset class, type, datatype, size
    stream.write(" | ".join([" Dataset ".center(col_width, "-")] * len(results)) + "\n")

    classes = ["Class:" + f"{"..."+result[0][-40:] if len(result[0]) > 43 else result[0]}".rjust(col_width-6) for result in results]
    stream.write(" | ".join(classes) + "\n")

    types = ["Type:" + f"{result[1][-40:] if len(result[1]) > 40 else result[1]}".rjust(col_width-5) for result in results]
    stream.write(" | ".join(types) + "\n")

    datatypes = [f"dtype:" + f"{result[3]}".rjust(col_width-6) for result in results]
    stream.write(" | ".join(datatypes) + "\n")

    sizes = ["Size:" + f"{result[2]:,}".rjust(col_width-5) for result in results]
    stream.write(" | ".join(sizes) + "\n")

    # Iterations, runtimes, times per iteration
    stream.write(" | ".join([" Time ".center(col_width, "-")] * len(results)) + "\n")

    iterations = ["Iterations:" + f"{iterations}".rjust(col_width-11) for x in range(len(results))]
    stream.write(" | ".join(iterations) + "\n")

    runtimes = ["Total time:" + f"{result[7]:.3f}s".rjust(col_width-11) for result in results]
    stream.write(" | ".join(runtimes) + "\n")

    times_per_iter = ["Time/iteration:" + f"{result[8]:.3f}s".rjust(col_width-15) for result in results]
    stream.write(" | ".join(times_per_iter) + "\n")

    # Writes top 5 longest running functions
    stream.write(" | ".join([" Functions ".center(col_width, "-")] * len(results)) + "\n")

    func_header = [f"FUNCTION NAME" + f"FILE:LINE - CUMTIME".rjust(col_width-13) for x in range(len(results))]
    stream.write(" | ".join(func_header) + "\n")
    for i in range(7):
        functions = " | ".join(result[9][i].ljust(col_width) for result in results)
        stream.write(functions + "\n")

    # Log to logger
    logger.info(f"\n{stream.getvalue()}")

    return


if __name__ == "__main__":
    main()