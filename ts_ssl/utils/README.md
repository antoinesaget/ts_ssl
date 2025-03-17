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
