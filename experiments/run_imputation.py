import numpy as np
import torch
import tsl.datasets as tsl_datasets
from neptune.utils import stringify_unsupported
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from tsl import logger
from tsl.data import SpatioTemporalDataset, SpatioTemporalDataModule
from tsl.data.preprocessing import scalers
from tsl.experiment import Experiment, NeptuneLogger
from tsl.metrics import torch_metrics
from tsl.nn import models as tsl_models

from lib.nn import baselines
from lib.nn.engines import MissingDataPredictor
from lib.utils import find_devices, add_missing_values, suppress_known_warnings



def model_class(model_str):
    #Baseline_imputation method:###
    if model_str == 'rnni':
        model = baselines.RNNIPredictionModel
    elif model_str == 'grin':
        model = baselines.GRINPredictionModel
    elif model_str == 'spin-h':
        model = baselines.SPINHierarchicalPredictionModel
    elif model_str == 'grud':
        model = baselines.GRUDModel
    # Forecasting models  ###############################################
    elif model_str == 'rnn':
        model = tsl_models.RNNModel
    elif model_str == 'dcrnn':
        model = tsl_models.DCRNNModel
    else:
        raise NotImplementedError(f'Model "{model_str}" not available.')

    return model



def get_dataset(dataset_cfg):
    name: str = dataset_cfg.name
    if name.startswith('la'):
        dataset = tsl_datasets.MetrLA(impute_zeros=True)
    elif name.startswith('bay'):
        dataset = tsl_datasets.PemsBay()
    elif name.startswith('air'):
        dataset = tsl_datasets.AirQuality(small=name[:5] == 'air36',
                                          impute_nans=False)
    elif name.startswith('Large'):
        years = [2017, 2018, 2019]
        dataset = tsl_datasets.LargeST(year = years,
                                       subset = next((s for s in ["GLA", "GBA", "SD"] if s in name), "CA"),  
                                       imputation_mode = "nearest")
    else:
        raise ValueError(f"Dataset {name} not present.")
    
    






