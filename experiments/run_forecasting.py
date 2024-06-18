import os

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

from lib.nn import baselines
from lib.utils import (find_devices,
                       add_missing_values,
                       suppress_known_warnings)
from lib.utils.statistics import prediction_dataframe_v3


class CombineImputations:
    def __init__(self, ordered_lags, mean_covariate, std_covariate):
        self.ordered_lags = ordered_lags
        self.mean_covariate = mean_covariate
        self.std_covariate = std_covariate

    def __call__(self, data):
        ## Combine the imputations
        scaler = data.transform['x']

        mean = torch.cat(
            [data.pop(f'mean/lag_{lag}')[..., i:i + 1, :, :]
             for i, lag in enumerate(self.ordered_lags)],
            dim=-3
        )
        sd = torch.cat(
            [data.pop(f'sd/lag_{lag}')[..., i:i + 1, :, :]
             for i, lag in enumerate(self.ordered_lags)],
            dim=-3
        )
        mean = scaler.transform(mean)

        # replace covariates with correct imputation mean and sd
        if self.mean_covariate and self.std_covariate:
            data.u[..., -2:] = torch.cat([sd, mean], dim=-1)
        elif self.mean_covariate:
            data.u[..., -1:] = mean
        elif self.std_covariate:
            data.u[..., -1:] = sd

        # impute missing values in x with the correct mean
        data.x = torch.where(data.input_mask, data.x, mean)
        return data


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
        dataset.target.fillna(0, inplace=True)
    elif name.startswith('LargeST'):
        dataset = tsl_datasets.LargeST(**dataset_cfg.hparams)
    else:
        raise ValueError(f"Dataset {name} not present.")

    # adjacency matrix
    adj = dataset.get_connectivity(**dataset_cfg.connectivity)
    # original mask
    mask = dataset.get_mask().copy()  # [time, node, feature]
    # new mask missing values:
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
    return dataset, adj, mask


def run(cfg: DictConfig):
    dataset, adj, original_mask = get_dataset(cfg.dataset)
    mask = dataset.mask
    data = dataset.dataframe()
    masked_data = data.where(mask.reshape(mask.shape[0], -1), np.nan)

    imputation_window = cfg.get('imputation_window', 24)

    custom_transform = None

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

        # Perform aggregation
        if cfg.imputation_model.name in ["birnni", "grin", "rnni"]:
            # New code
            lags = list(range(cfg.window, 0, -1))
            results_lag = prediction_dataframe_v3(
                combined_df,
                imputation_window=imputation_window,
                lags=lags + [imputation_window],
                aggregate_by=['mean', 'sd']
            )
            df_agg_mean = results_lag['mean'][imputation_window]
            df_agg_std = results_lag['sd'][imputation_window]
            df_agg_std = df_agg_std.fillna(0)

            custom_transform = CombineImputations(
                ordered_lags=lags,
                mean_covariate=cfg.dataset.covariates.mean,
                std_covariate=cfg.dataset.covariates.std,
            )
        else:
            df_agg_mean = dataframes[0]
            df_agg_std = pd.DataFrame(0, index=df_agg_mean.index,
                                      columns=df_agg_mean.columns)

    # covariates
    u = []
    if cfg.dataset.covariates.year:
        u.append(dataset.datetime_encoded('year').values)
    if cfg.dataset.covariates.day:
        u.append(dataset.datetime_encoded('day').values)
    if cfg.dataset.covariates.weekday:
        u.append(dataset.datetime_onehot('weekday').values)
    if cfg.dataset.covariates.mask:
        u.append(mask.astype(np.float32))
    if cfg.imputation_model.name != "none":
        if cfg.dataset.covariates.std:
            u.append(df_agg_std.values[..., None])
        if cfg.dataset.covariates.mean:
            u.append(df_agg_mean.values[..., None])

    # covariates union
    assert len(u)
    # ensure that all covariates have the same dimensionality
    # by expanding the one with lower dimension
    ndim = max(u_.ndim for u_ in u)
    u = np.concatenate([
        np.repeat(u_[:, None], dataset.n_nodes, 1)
        if u_.ndim < ndim else u_
        for u_ in u],
        axis=-1
    )

    if cfg.imputation_model.name != "none":
        # from the experiment filling the missing values using directly the imputed values
        data = masked_data.combine_first(df_agg_mean)
    else:
        # Fill nan with Last Observation Carried Forward
        data = masked_data.ffill().bfill()

    covariates = dict(u=u)
    # Add mean and std as covariates for every lag, i.e., distance from
    # the prediction horizon
    if cfg.imputation_model.name in ["birnni", "grin", "rnni"]:
        for method in ['mean', 'sd']:
            for lag in results_lag[method].keys():
                if lag != imputation_window:
                    df_lag = results_lag[method][lag]
                    df_lag = df_lag.combine_first(
                        results_lag[method][imputation_window])
                    covariates[f'{method}/lag_{lag}'] = df_lag

    # TODO: Al momento sto aggiungengo le imputation calcolate come la media su
    #  tutto per calcolare gli scalers. Non è troppo corretto perchè ci sono
    #  pochi dati (gli ultimi di training) che hanno visto qualche valore in più
    #  sul futuro. Essendo pochi, si può trascurare al momento.
    torch_dataset = SpatioTemporalDataset(
        target=data,
        mask=dataset.mask,
        connectivity=adj,
        covariates=covariates,  # dict(u=u),
        transform=custom_transform,
        horizon=cfg.horizon,
        window=cfg.window,
        delay=0,
        stride=cfg.stride,
    )

    # Add mask to model's inputs as 'input_mask'
    torch_dataset.update_input_map(input_mask=['mask'])

    scaler_cfg = cfg.get('scaler')
    if scaler_cfg is not None:
        scale_axis = (0,) if scaler_cfg.axis == 'node' else (0, 1)
        scaler_cls = getattr(scalers, f'{scaler_cfg.method}Scaler')
        transform = dict(target=scaler_cls(axis=scale_axis))
    else:
        transform = None

    dm = SpatioTemporalDataModule(
        dataset=torch_dataset,
        scalers=transform,
        splitter=dataset.get_splitter(**cfg.dataset.splitting),
        mask_scaling=cfg.imputation_model.name == "none",
        batch_size=cfg.batch_size,
        workers=cfg.workers,
    )
    dm.setup()

    # get the model
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

    # logger

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

    # ##Training

    early_stop_callback = EarlyStopping(monitor='val_mae',
                                        patience=cfg.patience,
                                        mode='min')

    checkpoint_callback = ModelCheckpoint(dirpath=cfg.run.dir,
                                          save_top_k=1,
                                          monitor="val_mae",
                                          mode='min')

    trainer = Trainer(max_epochs=cfg.epochs,
                      limit_train_batches=cfg.train_batches,
                      default_root_dir=cfg.run.dir,
                      logger=exp_logger,
                      accelerator='gpu' if torch.cuda.is_available() else 'cpu',
                      devices=find_devices(1),
                      gradient_clip_val=cfg.grad_clip_val,
                      callbacks=[early_stop_callback, checkpoint_callback])

    trainer.fit(predictor,
                train_dataloaders=dm.train_dataloader(),
                val_dataloaders=dm.val_dataloader())

    # testing

    predictor.load_model(checkpoint_callback.best_model_path)

    predictor.freeze()
    result = checkpoint_callback.best_model_score.item()

    trainer.test(predictor, datamodule=dm)

    ########################################
    # Test on unmasked data                #
    ########################################

    # Restore original mask with no missing values
    torch_dataset.set_mask(original_mask)
    # Restore original target with no missing values
    torch_dataset.set_data(dataset.numpy())

    # Restore x as input with injected missing values
    torch_dataset.add_covariate('x',
                                data,
                                't n f',
                                add_to_input_map=True,
                                preprocess=True)
    # Restore input_mask as mask with injected missing values
    torch_dataset.add_covariate('input_mask',
                                mask,
                                't n f',
                                add_to_input_map=True)
    # Scale again the target
    torch_dataset.add_scaler('x', torch_dataset.scalers['target'])

    from torchmetrics import MetricCollection
    predictor.test_metrics = MetricCollection(
        metrics={
            k: predictor._check_metric(m)
            for k, m in log_metrics.items()
        },
        prefix='test_',
        postfix='_unmasked',
    )
    trainer.test(predictor, dataloaders=dm.test_dataloader())

    if exp_logger is not None:
        exp_logger.finalize('success')

    return result


if __name__ == '__main__':
    suppress_known_warnings()
    exp = Experiment(run_fn=run,
                     config_path='../config/',
                     config_name='default_forecasting')
    res = exp.run()
    logger.info(res)
