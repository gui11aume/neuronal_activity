# neuronal_activity
Use BERT to model neuronal activity

## Setup Instructions

1. Clone this repository and navigate to the project directory:
   ```
   git clone https://github.com/your-username/neuronal_activity.git
   cd neuronal_activity
   ```

2. Install the required dependencies (this installs DVC as well):
   ```
   pip install -r requirements.txt
   ```

3. Enable DVC auto-staging (this should be done before any other operations):
   ```
   dvc config core.autostage true
   ```

## Configuration

You can modify the training parameters by editing the `config.yaml` file. This file contains various hyperparameters and settings for the model training process.

## Training

To start the training process, simply run:
```
make train
```

This command will execute the DVC pipeline defined in `dvc.yaml`, which includes data preparation and model training stages.

## Additional Information

- The `dvc.yaml` file defines the pipeline stages and their dependencies.
- Training logs and model outputs will be saved in the `logs` and `models` directories respectively.
- Use `dvc repro` to rerun the pipeline if you've made changes to the code or data.
