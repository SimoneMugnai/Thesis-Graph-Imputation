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


def get_model_class(model_str):
    # Your models  ######################################################
    if model_str.startswith('mymodel'):
        # model = models.MyModel
        pass
    # Baseline models  ##################################################
    elif model_str == 'rnni':
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
    # Get the dataset
    name: str = dataset_cfg.name
    if name.startswith('air'):
        dataset = tsl_datasets.AirQuality(small=name[:5] == 'air36',
                                          impute_nans=False)
        dataset.target.fillna(0, inplace=True)
    elif name.startswith('la'):
        dataset = tsl_datasets.MetrLA(impute_zeros=True)
    elif name.startswith('bay'):
        dataset = tsl_datasets.PemsBay()
    elif name.startswith('Large'):
        years = [2019]
        dataset = tsl_datasets.LargeST(year = years,
                                       subset = next((s for s in ["GLA", "GBA", "SD"] if s in name), "CA"),  
                                       imputation_mode = "nearest")
    else:
        raise ValueError(f"Dataset {name} not available.")
    adj = dataset.get_connectivity(**dataset_cfg.connectivity)
    # Get original mask
    mask = dataset.get_mask().copy()  # [time, node, feature]
    # Add missing values to dataset
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
    ########################################
    # Get Dataset                          #
    ########################################
    dataset, adj, original_mask = get_dataset(cfg.dataset)

    # Get mask
    mask = dataset.mask

    # Get covariates
    u = []
    if cfg.dataset.covariates.year:
        u.append(dataset.datetime_encoded('year').values)
    if cfg.dataset.covariates.day:
        u.append(dataset.datetime_encoded('day').values)
    if cfg.dataset.covariates.weekday:
        u.append(dataset.datetime_onehot('weekday').values)
    if cfg.dataset.covariates.mask:
        u.append(mask.astype(np.float32))

    # Concatenate covariates
    assert len(u)
    ndim = max(u_.ndim for u_ in u)
    u = np.concatenate([
        np.repeat(u_[:, None], dataset.n_nodes, 1)
        if u_.ndim < ndim else u_
        for u_ in u],
        axis=-1
    )

    # Get data and set missing values to nan
    data = dataset.dataframe()
    masked_data = data.where(mask.reshape(mask.shape[0], -1), np.nan)

    # Fill nan with Last Observation Carried Forward
    data = masked_data.ffill().bfill()

    torch_dataset = SpatioTemporalDataset(target=data,
                                          mask=mask,
                                          covariates=dict(u=u),
                                          connectivity=adj,
                                          horizon=cfg.horizon,
                                          window=cfg.window,
                                          stride=cfg.stride)

    # Add mask to model's inputs as 'input_mask'
    torch_dataset.update_input_map(input_mask=['mask'])

    # Scale input features
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
        mask_scaling=True,
        batch_size=cfg.batch_size,
        workers=cfg.workers)
    dm.setup()

    if cfg.model.name == 'grud':
        x_mean = (masked_data.iloc[dm.train_slice].mean().values.reshape(
            dataset.n_nodes, dataset.n_channels))
        dm.torch_dataset.add_covariate('x_mean', x_mean)

    ########################################
    # Create model                         #
    ########################################

    model_cls = get_model_class(cfg.model.name)

    d_exog = torch_dataset.input_map.u.shape[-1] if 'u' in torch_dataset else 0

    model_kwargs = dict(n_nodes=torch_dataset.n_nodes,
                        input_size=torch_dataset.n_channels,
                        mask_size=torch_dataset.n_channels,
                        exog_size=d_exog,
                        output_size=torch_dataset.n_channels,
                        horizon=torch_dataset.horizon)

    model_cls.filter_model_args_(model_kwargs)
    model_kwargs.update(cfg.model.hparams)

    ########################################
    # predictor                            #
    ########################################

    loss_fn = torch_metrics.MaskedMAE()
    if cfg.imputation_loss:
        imputation_loss_fn = torch_metrics.MaskedMAE()
    else:
        imputation_loss_fn = None

    log_metrics = {
        'mae': torch_metrics.MaskedMAE(),
        'mse': torch_metrics.MaskedMSE(),
        'mre': torch_metrics.MaskedMRE()
    }

    if cfg.get('lr_scheduler') is not None:
        scheduler_class = getattr(torch.optim.lr_scheduler,
                                  cfg.lr_scheduler.name)
        scheduler_kwargs = dict(cfg.lr_scheduler.hparams)
    else:
        scheduler_class = scheduler_kwargs = None

    # setup predictor
    predictor = MissingDataPredictor(
        model_class=model_cls,
        model_kwargs=model_kwargs,
        optim_class=getattr(torch.optim, cfg.optimizer.name),
        optim_kwargs=dict(cfg.optimizer.hparams),
        loss_fn=loss_fn,
        metrics=log_metrics,
        scheduler_class=scheduler_class,
        scheduler_kwargs=scheduler_kwargs,
        scale_target=False if scaler_cfg is None else scaler_cfg.scale_target,
        whiten_prob=cfg.whiten_prob,
        log_lr=cfg.get('log_lr', True),
        log_grad_norm=cfg.get('log_grad_norm', False),
        imputation_loss_fn=imputation_loss_fn,
        imputation_loss_weight=cfg.imputation_loss_weight,
        imputation_warm_up=cfg.imputation_warm_up,
    )

    ########################################
    # logging options                      #
    ########################################

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
    else:
        raise NotImplementedError("Logger backend not supported.")

    ########################################
    # training                             #
    ########################################

    early_stop_callback = EarlyStopping(monitor='val_mae',
                                        patience=cfg.patience,
                                        mode='min')

    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg.run.dir,
        save_top_k=1,
        monitor='val_mae',
        mode='min',
    )

    trainer = Trainer(
        max_epochs=cfg.epochs,
        limit_train_batches=cfg.train_batches,
        default_root_dir=cfg.run.dir,
        logger=exp_logger,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=find_devices(1),
        gradient_clip_val=cfg.grad_clip_val,
        callbacks=[early_stop_callback, checkpoint_callback])

    load_model_path = cfg.get('load_model_path')
    if load_model_path is not None:
        predictor.load_model(load_model_path)
    else:
        trainer.fit(predictor,
                    train_dataloaders=dm.train_dataloader(),
                    val_dataloaders=dm.val_dataloader())
        predictor.load_model(checkpoint_callback.best_model_path)

    predictor.freeze()

    result = checkpoint_callback.best_model_score.item()

    ########################################
    # testing                              #
    ########################################

    trainer.test(predictor, dataloaders=dm.test_dataloader())

    ########################################
    # Test on unmasked data                #
    ########################################

    # Restore original mask
    torch_dataset.set_mask(original_mask)
    # Restore target
    torch_dataset.set_data(dataset.numpy())
    # Add data with imputations as input
    torch_dataset.add_covariate('x',
                                data,
                                't n f',
                                add_to_input_map=True,
                                preprocess=True)
    # Add again scaler for input
    torch_dataset.add_scaler('x', torch_dataset.scalers['target'])
    # Add mask only to mask input
    torch_dataset.add_covariate('input_mask',
                                mask,
                                't n f',
                                add_to_input_map=True)

    from torchmetrics import MetricCollection
    predictor.test_metrics = MetricCollection(metrics={
        k: predictor._check_metric(m)
        for k, m in log_metrics.items()
    },
        prefix='test_',
        postfix='_unmasked')
    trainer.test(predictor, dataloaders=dm.test_dataloader())

    if exp_logger is not None:
        exp_logger.finalize('success')

    return result


if __name__ == '__main__':
    suppress_known_warnings()
    exp = Experiment(run_fn=run,
                     config_path='../config/',
                     config_name='default')
    res = exp.run()
    logger.info(res)
