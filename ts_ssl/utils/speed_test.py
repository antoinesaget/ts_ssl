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


@hydra.main(config_path="util_config", config_name="config", version_base="1.3")
def main(config):
    # Get config parameters
    data = config.dataset.ssl_data
    iterations = config.training.iterations
    # Load datasets
    dataset_arrow = load_dataset(path=data["path"], split=data["split"]).with_format(
        "torch", columns=["x", "y"], dtype=torch.float16)
    
    dataset_SSL = SSLGroupedTimeSeriesDataset(
        data=config.dataset.ssl_data,
        n_samples_per_group=config.training.n_samples_per_group,
        percentiles=config.dataset.percentiles,
        config=config.augmentations,
        normalize_data=config.dataset.normalize,
        dataset_type=config.dataset.dataset_type,
    )

    # Make dataloaders
    loader_arrow = DataLoader(dataset=dataset_arrow, batch_size=config.training.batch_size, num_workers=config.training.num_workers)
    loader_SSL = DataLoader(dataset=dataset_SSL,batch_size=config.training.batch_size, num_workers=config.training.num_workers)

    # Time iterations of traversing datasets
    data_arrow = time_runs(loader_arrow, iterations)
    data_SSL = time_runs(loader_SSL, iterations)

    #logging.getLogger(__name__).info(f"data_arrow: {type(data_arrow[2].func_profiles.values())}")
    compare_runs(data_arrow, data_SSL)


def iterate_dataset(dataset: Dataset, batch_size: int, num_workers: int):
    '''
    Creates a DataLoader instance and iterates through the dataset with given parameters
    '''
    loader = DataLoader(dataset=dataset, batch_size=batch_size, num_workers=num_workers)
    for item in loader:
        len(item)


def iterate_dataset(loader: DataLoader):
    '''
    Takes a DataLoader instance and iterates through the dataset with given parameters
    '''
    for item in loader:
        len(item["x"])


def time_runs(loader: DataLoader, num_times: int):
    '''
    Iterates through an entire dataset using DataLoader <num_int> number of times. Timed using cProfile; statistics logged after iterations;

    Returns:
        - Tuple = (dataset type, num_times, StatsProfile of run)
    '''
    # Get logger and IO to store profile stats
    log = logging.getLogger(__name__)
    stats_stream = io.StringIO()

    # Using a profiler, iterate through the dataset <num_times> times
    with tqdm(total=(num_times*len(loader))) as pbar:
        with cProfile.Profile() as pr:
            for iteration in range(num_times):
                for item in loader:
                    len(item)
                    pbar.update(1)

    # Obtain the profiler stats, format, and print to IO 
    profile_stats = pstats.Stats(pr, stream=stats_stream).sort_stats("cumulative")
    profile_stats.print_stats(15)

    # Use IO to print stats to logger
    #log.info(f"Iterated {num_times} time(s) through dataset of size {len(loader.dataset)} with a batch size of {loader.batch_size} and {loader.num_workers} worker(s): \n\n{stats_stream.getvalue()}")
    
    return type(loader.dataset), num_times, pr

def present_stats(stats: StatsProfile):
    log = logging.getLogger(__name__)
    log.info(f"{stats.total_tt}")
    log.info(list(stats.func_profiles.values())[0])
    sorted_list = sorted(stats.func_profiles.items(), key=lambda x: x[1].cumtime, reverse=True)
    for key, val in sorted_list[:10]:
        log.info(f"{key[:10] + "..." if len(key) > 10 else key + " " * (13-len(key))}: {val.cumtime}")

    return

def compare_runs(run1: tuple[str, int, Profile], run2: tuple[str, int, Profile]):
    template = """----------------Iteration Speed Comparision----------------
          {r1name:.23}  | {r2name:.21}
Iterations: {r1iter:>21}  |{r2iter:>21}
Total Time: {r1time:>21}s |{r2time:>21}s
Time/Iter:  {r1avg:>21}s |{r2avg:>21}s
Longest Functions:
            {r1lgst:<15}{r1lgsttime:>6}s | {r2lgst:<15}{r2lgsttime:>6}s
    """
    
    storage = io.StringIO()
    r1stats = pstats.Stats(run1[2], stream=storage).get_stats_profile()
    r2stats = pstats.Stats(run2[2], stream=storage).get_stats_profile()

    r1type, r2type = str(run1[0]),str(run2[0])
    storage.write(f"------Comparing datasets of types {r1type[:25] + ".." if len(r1type) > 25 else r1type} and {r2type[:25] + ".." if len(r2type) > 25 else r2type}------\n")
    r1_functions_sorted, r2_functions_sorted = sorted(r1stats.func_profiles.items(), key=lambda x: x[1].cumtime,
                                                          reverse=True), sorted(r2stats.func_profiles.items(), key=lambda x: x[1].cumtime, reverse=True)
    r1tt, r2tt = r1stats.total_tt, r2stats.total_tt
    r1avg, r2avg = r1tt/run1[1], r2tt/run2[1]
    r1lgst, r1lgsttime = r1_functions_sorted[0][0], r1_functions_sorted[0][1].cumtime
    r2lgst, r2lgsttime = r2_functions_sorted[0][0], r2_functions_sorted[0][1].cumtime
    r1lgst, r1lgsttime = r1_functions_sorted[0][0], r1_functions_sorted[0][1].cumtime
    r2lgst, r2lgsttime = r2_functions_sorted[0][0], r2_functions_sorted[0][1].cumtime
    
    storage.write(template.format(r1name=r1type,
                                  r2name=r2type,
                                  r1iter=run1[1],
                                  r2iter=run2[1],
                                  r1time=r1tt,
                                  r2time=r2tt,
                                  r1avg=r1avg,
                                  r2avg=r2avg,
                                  r1lgst=r1lgst,
                                  r2lgst=r2lgst,
                                  r1lgsttime=r1lgsttime,
                                  r2lgsttime=r2lgsttime,
                                  ))



    logging.getLogger(__name__).info(storage.getvalue())
    


if __name__ == "__main__":
    main()