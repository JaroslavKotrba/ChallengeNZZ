# Analysis and Model Creation 

This repository contains the necessary steps to set up the environment, install required packages, and run the scripts for creating and analyzing models using a provided dataset.

## Installation

### Step 1: Copy the dataset
Copy the `dataset_propensity.json` file to the `data` folder.

### Step 2: Create and activate conda environment
Create a new Conda environment with Python 3.10:
```sh
conda create -n nzz python=3.10
conda activate nzz
```

### Step 3: Install libraries
Run in activated env
```sh
pip install pandas matplotlib seaborn plotly nbformat scikit-learn xgboost shap
```

### Step 4: Create clean dataset
Run ../analysis/analysis.py to create clean_data.csv
```sh
python analysis.py
```

### Step 5: Create model
Run ../src/main.py to create model_pipe.csv
```sh
python main.py
```