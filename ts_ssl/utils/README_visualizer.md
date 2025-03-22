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
The visualizer configuration is stored in a dedicated file `ts_ssl/config/visualizer/default.yaml` and is automatically loaded through the main project configuration system. This keeps the main config file clean while still allowing the visualizer to use the same dataset and augmentation configurations as the main project.

### Visualization Functions

The visualizer supports three main functions, controlled by the `visualizer.function` parameter:

#### 1. Plot Before and After Augmentation (`plotbeforeandafter`)

Plots one sample before and after data augmentations are applied.

```bash
# Basic example
python visualizer.py visualizer.function=plotbeforeandafter augmentations=masking

# Specify a sample index
python visualizer.py visualizer.function=plotbeforeandafter visualizer.sample=42

### Additional Configuration Options
- `visualizer.sample`: Specify which sample to visualize (default: random)
- `visualizer.overlay`: Whether to overlay before/after plots (default: true)
- `visualizer.bands`: List of spectral bands to display (default: all bands)
- `visualizer.seed`: Seed for random sample selection (default: None)
- `dataset.normalize`: Whether to normalize the data (recommended to set to false for visualization)

# Display plots overlaid instead of one above the other
python visualizer.py visualizer.function=plotbeforeandafter visualizer.overlay=true

# Specify which spectral bands to display
python visualizer.py visualizer.function=plotbeforeandafter visualizer.bands=[0,1,2]
```

#### 2. Plot Single Class Examples (`plotsingleclass`)
Plots 10 examples from a single specified class.

```bash
# Display 10 samples of class 5
python visualizer.py visualizer.function=plotsingleclass visualizer.classid=5
```

#### 3. Plot Multi-Class Examples (`plotmulticlass`)
Plots one example from each class in the dataset (20 total).

```bash
# Display one sample from each class
python visualizer.py visualizer.function=plotmulticlass
```

### Global Configuration Options
These options apply to all visualization functions:

- `dataset.normalize`: Whether to normalize the data (default: depends on dataset; recommended to set to false for visualization)
  ```bash
  python visualizer.py dataset.normalize=false
  ```

- `visualizer.save_plot`: Whether to save plots to files instead of displaying them (default: false)
  ```bash
  # Save the plot to the output directory instead of displaying it
  python visualizer.py visualizer.save_plot=true
  ```

### Output Files
When `visualizer.save_plot` is set to true, the plots will be saved with the following naming conventions:

- For `plotbeforeandafter`: `beforeafter_{augmentation_name}_{sample_index}.png`
- For `plotsingleclass`: `class_{class_id}_examples.png`
- For `plotmulticlass`: `multiclass_examples.png`

All plots are saved to the output directory specified in the main configuration.
