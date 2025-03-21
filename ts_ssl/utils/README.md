# Visualizer Utility

## Overview
The `visualizer.py` utility allows for visualizing data from datasets within the ts_ssl project. It provides three main visualization functions that can be executed via command line with Hydra configuration syntax.

## Usage
To use the visualizer, navigate to the `/ts_ssl/utils` directory and run:

```bash
python visualizer.py [configuration_options]
```

For example:
```bash
python visualizer.py dataset.normalize=false
```

## Configuration
The visualizer configuration is stored in a dedicated file `ts_ssl/config/visualizer.yaml` and is automatically loaded through the main project configuration system. This keeps the main config file clean while still allowing the visualizer to use the same dataset and augmentation configurations as the main project.

### Visualization Functions
The visualizer supports three main functions, controlled by the `visualizer.function` parameter:

1. **plotbeforeandafter**: Plots one sample specified by `visualizer.sample`, before and after data augmentations are applied.
   - Change the augmentation type using: `augmentations=combination|jittering|masking|resampling|resizing`
   - Example: `python visualizer.py visualizer.function=plotbeforeandafter augmentations=masking`

2. **plotsingleclass**: Plots one random sample of class specified by `visualizer.classid` (0-19).
   - Example: `python visualizer.py visualizer.function=plotsingleclass visualizer.classid=5`

3. **plotmulticlass**: Plots one random sample per class in dataset (20 total).
   - Example: `python visualizer.py visualizer.function=plotmulticlass`

### Additional Configuration Options
- `visualizer.sample`: Specify which sample to visualize (default: random)
- `visualizer.overlay`: Whether to overlay before/after plots (default: true)
- `visualizer.bands`: List of spectral bands to display (default: all bands)
- `visualizer.seed`: Seed for random sample selection (default: None)
- `dataset.normalize`: Whether to normalize the data (recommended to set to false for visualization)



# Dataloading Utility

## Overview
The `dataloader.py` utility allows for analysis of data retrieval times within the ts_ssl project. It provides four main comparison functions that can be executed via command line with Hydra configuration syntax. 

## Usage
To use the visualizer, navigate to the `/ts_ssl/utils` directory and run:

```bash
python speed_test.py [configuration_options]
```

For example:
```bash
python speed_test.py dataloaderutil.function=runaugmentations dataloaderutil.iterations=5
```

# Configuration
The dataloader util configuration is stored in a dedicated file `ts_ssl/config/dataloaderutil/default.yaml` and is automatically loaded through the main project configuration system. This keeps the main config file clean while still allowing the dataloader util to use the same dataset and augmentation configurations as the main project.

### Comparison Functions
The visualizer supports four main functions, controlled by the `dataloaderutil.function` parameter:

1. **runarrowmmap**: Times and compares the iterations of two datasets: one utilizing parquet arrow format, the other utilizing mmap format.
   - Change dataset parameters using: `python dataloader.py dataloaderutil.<mmap|huggingface>.<parameter>=<value>`
   - Example: `python speed_test.py dataloaderutil.function=runarrowmap dataloader.util.mmap.normalize=False`

2. **runaugmentations**: Times and compares the iterations of up to three datasets using augmentations specified in `dataloaderutil.augmentations` parameter.
   - Change augmentations using: `dataloaderutil.augmentations=[augmentation0,augmentation1,augmentation2]` with agumentation in `(combination|jittering|masking|resampling|resizing)`
   - Example: `python speed_test.py dataloaderutil.function=runaugmentations dataloaderutil.augmentations=[resampling,combination,masking]`

3. **runnumworkers**: Times and compares the iterations of a dataset using up to three dataloaders with <num_workers> specified in `dataloaderutil.num_workers` parameter.
   - Change number of workers to compare using: `dataloaderutil.num_workers=[nw0,nw1,nw2]` with `nw > 0`
   - Example: `python speed_test.py dataloaderutil.function=runnumworkers dataloaderutil.num_workers=[4,8,16]`
   - Note: see https://pytorch.org/docs/stable/data.html#single-and-multi-process-data-loading for parameter limitations

4. **runbatchsize**: Times and compares the iterations of a dataset using up to three dataloaders with <batch_size> specified in `dataloaderutil.batch_size` parameter.
   - Change batch sizes to compare using: `dataloaderutil.batch_sizes=[bs0,bs1,bs2]` with `bs > 0`
   - Example: `python speed_test.py dataloaderutil.function=runbatchsize dataloaderutil.num_workers=[512,1024,2048]`

### Additional Configuration Options
- `dataloaderutil.iterations`: Specify how many times a dataset should be traversed in a run (default: 10)
- `dataloaderutil.dataset_type`: Specify which dataset type will be used; ignored when using `runarrowmmap` (default: mmap)
- `num_workers`: Specify number of workers for multi-process data loading; ignored when using `runnumworkers` (default: 4)
- `training.batch_size`: Specify batch size for data loading; ignored when using `runbatchsize` (default: 1024)