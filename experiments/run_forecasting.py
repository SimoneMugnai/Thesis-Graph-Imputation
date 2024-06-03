import numpy as np
import pandas as pd
import torch
import tsl.datasets as tsl_datasets
from neptune.utils import stringify_unsupported
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from tsl import logger
from tsl.data import SpatioTemporalDataModule, SpatioTemporalDataset
from tsl.data.preprocessing import scalers
from tsl.engines import Predictor
from tsl.experiment import Experiment, NeptuneLogger
from tsl.metrics import torch as torch_metrics
from tsl.nn import models as tsl_models
import os

from lib.nn import baselines
from lib.utils import find_devices, add_missing_values, suppress_known_warnings, \
    prediction_dataframe_v2, calculate_residuals


def get_model_class(model_str):
    # Spatiotemporal Models ###################################################
    if model_str == 'dcrnn':
        model = tsl_models.DCRNNModel  # (Li et al., ICLR 2018)
    elif model_str == 'gwnet':
        model = tsl_models.GraphWaveNetModel  # (Wu et al., IJCAI 2019)
    elif model_str == 'evolvegcn':
        model = tsl_models.EvolveGCNModel  # (Pereja et al., AAAI 2020)
    elif model_str == 'agcrn':
        model = tsl_models.AGCRNModel  # (Bai et al., NeurIPS 2020)
    # elif model_str == 'grugcn':
    #     model = tsl_models.GRUGCNModel  # (Guo et al., ICML 2022)
    # elif model_str == 'gatedgn':
    #     model = tsl_models.GatedGraphNetworkModel  # (Satorras et al., 2022)
    # elif model_str == 'stcn':
    #     model = tsl_models.STCNModel
    elif model_str == 'transformer':
        model = tsl_models.TransformerModel
    elif model_str == 'tts_imp':
        model = baselines.TimeThenGraphIsoModel
    elif model_str == 'tts_amp':
        model = baselines.TimeThenGraphAnisoModel
    elif model_str == 'tas_imp':
        model = baselines.TimeAndGraphIsoModel
    elif model_str == 'tas_amp':
        model = baselines.TimeAndGraphAnisoModel
    # Predictors with missing data  ###########################################
    elif model_str == 'rnni':
        model = baselines.RNNIPredictionModel
    elif model_str == 'grin':
        model = baselines.GRINPredictionModel
    elif model_str == 'mtan':
        model = baselines.MTANPredictionModel
    elif model_str == 'spin-h':
        model = baselines.SPINHierarchicalPredictionModel
    elif model_str == 'grud':
        model = baselines.GRUDModel
    # Temporal Models #########################################################
    elif model_str == 'ar':
        model = tsl_models.ARModel
    elif model_str == 'var':
        model = tsl_models.VARModel
    elif model_str == 'rnn':
        model = tsl_models.RNNModel
    else:
        raise NotImplementedError(f'Model "{model_str}" not available.')
    return model


def get_dataset(dataset_cfg):
    name: str = dataset_cfg.name
    if name.startswith('la'):
        dataset = tsl_datasets.MetrLA(impute_zeros=True)
        print(dataset)
    elif name.startswith('bay'):
        dataset = tsl_datasets.PemsBay()
    elif name.startswith('air'):
        dataset = tsl_datasets.AirQuality(small=name[:5] == 'air36',
                                          impute_nans=False)
    elif name.startswith('Large'):
        years = [2017, 2018, 2019]
        dataset = tsl_datasets.LargeST(year = years,
                                       subset = next((s for s in ["GLA", "GBA", "SD"] if s in name), "SD"),  
                                       imputation_mode = "nearest")
    else:
        raise ValueError(f"Dataset {name} not present.")
    
    #adjacency matrix 
    adj = dataset.get_connectivity(**dataset_cfg.connectivity)
    #original mask
    #new mask missing values:
    if dataset_cfg.missing.name != 'normal':
        add_missing_values(
            dataset,
            p_fault=dataset_cfg.missing.p_fault,
            p_noise=dataset_cfg.missing.p_noise,
            min_seq=dataset_cfg.missing.min_seq,
            max_seq=dataset_cfg.missing.max_seq,
            p_propagation=dataset_cfg.missing.get('p_propagation', 0),
            connectivity=adj,
            propagation_hops=dataset_cfg.missing.get('propagation_hops', 0),
            seed=dataset_cfg.missing.seed)
        dataset.set_mask(dataset.training_mask)
    return dataset, adj

def run(cfg: DictConfig):
    dataset,adj = get_dataset(cfg.dataset)
    mask = dataset.mask
    data = dataset.dataframe()
    masked_data = data.where(mask.reshape(mask.shape[0], -1), np.nan)


    if cfg.imputation_model.name != "none":  
        hdf5_files = os.listdir(cfg.dir_imputation)
        # Initialize an empty list to store dataframes
        dataframes = []
    
        # Loop through each HDF5 file and load the dataset
        for hdf5_file in hdf5_files:
            file_path = os.path.join(cfg.dir_imputation, hdf5_file)
            df_imputed = pd.read_hdf(file_path, key='imputed_dataset')
            dataframes.append(df_imputed)

        # Concatenate all dataframes into a single dataframe
        combined_df = pd.concat(dataframes)
        #Perform aggregation
        aggr_by = ['mean', 'sd']
        results = prediction_dataframe_v2(combined_df.values, combined_df.index, combined_df.columns, aggregate_by=aggr_by)
        df_agg_mean = results['mean']
        df_agg_std = results['sd']
        df_agg_std = df_agg_std.fillna(0)
        residuals = calculate_residuals(masked_data, df_agg_mean)
        #set to 0 the std when I have missing values in the original dataset
        df_agg_std = df_agg_std.where(residuals != 0, 0)





    #covariates
    u = []
    if cfg.dataset.covariates.year:
        u.append(dataset.datetime_encoded('year').values)
    if cfg.dataset.covariates.day:
        u.append(dataset.datetime_encoded('day').values)
    if cfg.dataset.covariates.weekday:
        u.append(dataset.datetime_onehot('weekday').values)
    if cfg.dataset.covariates.mask:
        u.append(mask.astype(np.float32))
    if cfg.dataset.covariates.std:
        u.append(df_agg_std.values[...,None])
    if cfg.dataset.covariates.residual:
        u.append(residuals.values[...,None])
    
    # covariates union
    assert len(u)
    #ensure that all covariates have the same dimensionalities
    #by expanding the one with lower dimension
    ndim = max(u_.ndim for u_ in u)
    u = np.concatenate([
        np.repeat(u_[:, None], dataset.n_nodes, 1)
        if u_.ndim < ndim else u_
        for u_ in u],
        axis=-1
    )

    
    if cfg.imputation_model.name != "none":
        #from the experiment filling the missing values using directly the imputed values
        data = masked_data.combine_first(df_agg_mean)
    else:
        # Fill nan with Last Observation Carried Forward
        data = masked_data.ffill().bfill()


    torch_dataset = SpatioTemporalDataset(
        target=data,
        mask=dataset.mask,
        connectivity=adj,
        covariates=dict(u=u),
        horizon=cfg.horizon,
        window=cfg.window,
        delay= 0,
        stride=cfg.stride,
    )

    scaler_cfg = cfg.get('scaler')
    if scaler_cfg is not None:
        scale_axis = (0,) if scaler_cfg.axis == 'node' else (0, 1)
        scaler_cls = getattr(scalers, f'{scaler_cfg.method}Scaler')
        transform = dict(target=scaler_cls(axis=scale_axis))
    else:
        transform = None

    dm = SpatioTemporalDataModule(dataset = torch_dataset,
                                    scalers = transform,
                                    splitter = dataset.get_splitter(**cfg.dataset.splitting),
                                    mask_scaling = False,
                                    batch_size = cfg.batch_size,
                                    workers = cfg.workers )
    dm.setup()

    

    #predictors

    #get the model
    model_cls = get_model_class(cfg.model.name)
    d_exog = torch_dataset.input_map.u.shape[-1] if 'u' in torch_dataset else 0

    model_kwargs = dict(n_nodes=torch_dataset.n_nodes,
                        input_size=torch_dataset.n_channels,
                        output_size=torch_dataset.n_channels,
                        horizon=torch_dataset.horizon,
                        exog_size=d_exog)

    model_cls.filter_model_args_(model_kwargs)
    model_kwargs.update(cfg.model.hparams)

    loss_fn = torch_metrics.MaskedMAE()

    log_metrics = {
        'mae': torch_metrics.MaskedMAE(),
        'mse': torch_metrics.MaskedMSE(),
        'mape': torch_metrics.MaskedMAPE()
    }

    if cfg.lr_scheduler is not None:
        scheduler_class = getattr(torch.optim.lr_scheduler,
                                  cfg.lr_scheduler.name)
        scheduler_kwargs = dict(cfg.lr_scheduler.hparams)
    else:
        scheduler_class = scheduler_kwargs = None

    
    predictor = Predictor(
        model_class=model_cls,
        model_kwargs=model_kwargs,
        optim_class=getattr(torch.optim, cfg.optimizer.name),
        optim_kwargs=dict(cfg.optimizer.hparams),
        loss_fn=loss_fn,
        metrics=log_metrics,
        scheduler_class=scheduler_class,
        scheduler_kwargs=scheduler_kwargs,
        scale_target=False if scaler_cfg is None else scaler_cfg.scale_target,
    )


    #logger

    run_args = exp.get_config_dict()
    run_args['model']['trainable_parameters'] = predictor.trainable_parameters
    run_args = stringify_unsupported(run_args)

    if cfg.get('logger') is None:
        exp_logger = None
    elif cfg.logger.backend == 'neptune':
        exp_logger = NeptuneLogger(project_name=cfg.logger.project,
                                   experiment_name=cfg.run.name,
                                   save_dir=cfg.run.dir,
                                   tags=cfg.tags,
                                   params=run_args,
                                   debug=cfg.logger.offline)
        exp_logger.log_artifact(os.path.join(exp.run_dir, "config.yaml"))
    else:
        raise NotImplementedError("Logger backend not supported.")
    
    ##Training

    early_stop_callback = EarlyStopping(monitor = 'val_mae',
                                        patience = cfg.patience,
                                        mode = 'min'
                                        )
    

    checkpoint_callback = ModelCheckpoint(dirpath = cfg.run.dir,
                                          save_top_k=1,
                                          monitor = "val_mae",
                                          mode = 'min'
                                          )
    
    trainer = Trainer(max_epochs = cfg.epochs,
                      limit_train_batches = cfg.train_batches,
                      default_root_dir = cfg.run.dir,
                      logger = exp_logger,
                      accelerator='gpu' if torch.cuda.is_available() else 'cpu',
                      devices=find_devices(1),
                      gradient_clip_val = cfg.grad_clip_val,
                      callbacks=[early_stop_callback, checkpoint_callback])
    
    trainer.fit(predictor, train_dataloaders = dm.train_dataloader(),
                    val_dataloaders= dm.val_dataloader())
    

    #testing

    predictor.load_model(checkpoint_callback.best_model_path)

    predictor.freeze()
    trainer.test(predictor, datamodule=dm)

if __name__ == '__main__':
    suppress_known_warnings()
    exp = Experiment(run_fn=run,
                    config_path='../config/',
                    config_name='default_forecasting')
    res = exp.run()
    logger.info(res)
