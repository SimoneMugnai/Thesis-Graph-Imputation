import os
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import tsl.datasets as tsl_datasets
from neptune.utils import stringify_unsupported
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from tsl import logger
from tsl.data import SpatioTemporalDataModule, ImputationDataset
from tsl.data.preprocessing import scalers
from tsl.engines import Imputer
from tsl.experiment import Experiment, NeptuneLogger
from tsl.metrics import torch_metrics
from tsl.nn import models as tsl_models
from tsl.transforms import MaskInput

from lib.nn import baselines
from lib.utils import find_devices, add_missing_values, suppress_known_warnings


def get_model_class(model_str):
    # Baseline_imputation method:###
    if model_str == 'rnni':
        model = tsl_models.RNNImputerModel
    elif model_str == 'birnni':
        model = tsl_models.BiRNNImputerModel
    elif model_str == 'grin':
        model = baselines.GRINModel  # override to handle 2dim covariates
    elif model_str == 'spin':
        model = tsl_models.SPINModel
    elif model_str == 'spin-h':
        model = tsl_models.SPINHierarchicalModel
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
        dataset = tsl_datasets.LargeST(year=years,
                                       subset=next((s for s in ["GLA", "GBA", "SD"] if s in name), "SD"),
                                       imputation_mode="nearest")
    else:
        raise ValueError(f"Dataset {name} not present.")

    # adjacency matrix
    adj = dataset.get_connectivity(**dataset_cfg.connectivity)

    # Add missing values in 'eval_mask': now 'training_mask' becomes
    # the new mask
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
    return dataset, adj


def run(cfg: DictConfig):
    dataset, adj = get_dataset(cfg.dataset)
    # covariates
    u = []
    if cfg.dataset.covariates.year:
        u.append(dataset.datetime_encoded('year').values)
    if cfg.dataset.covariates.day:
        u.append(dataset.datetime_encoded('day').values)
    if cfg.dataset.covariates.weekday:
        u.append(dataset.datetime_onehot('weekday').values)

    # covariates union
    assert len(u)
    # ensure that all covariates have the same dimensionalities
    # by expanding the one with lower dimension
    ndim = max(u_.ndim for u_ in u)
    u = np.concatenate([
        np.repeat(u_[:, None], dataset.n_nodes, 1)
        if u_.ndim < ndim else u_
        for u_ in u],
        axis=-1
    )

    # dataset in torch
    torch_dataset = ImputationDataset(target=dataset.dataframe(),
                                      mask=dataset.training_mask,
                                      eval_mask=dataset.eval_mask,
                                      covariates=dict(u=u),
                                      transform=MaskInput(),
                                      connectivity=adj,
                                      window=cfg.window,
                                      stride=cfg.stride)

    # scale input
    scaler_cfg = cfg.get("scaler")
    if scaler_cfg is not None:
        scale_axis = (0,) if scaler_cfg.axis == "node" else (0, 1)
        scaler_cls = getattr(scalers, f'{scaler_cfg.method}Scaler')
        transform = dict(target=scaler_cls(axis=scale_axis))
    else:
        transform = None

    dm = SpatioTemporalDataModule(dataset=torch_dataset,
                                  scalers=transform,
                                  splitter=dataset.get_splitter(
                                      **cfg.dataset.splitting),
                                  mask_scaling=True,
                                  batch_size=cfg.batch_size,
                                  workers=cfg.workers)
    dm.setup()

    # get the model
    model_cls = get_model_class(cfg.model.name)
    d_exog = torch_dataset.input_map.u.shape[-1] if 'u' in torch_dataset else 0

    model_kwargs = dict(n_nodes=torch_dataset.n_nodes,
                        input_size=torch_dataset.n_channels,
                        exog_size=d_exog,
                        output_size=torch_dataset.n_channels)

    model_cls.filter_model_args_(model_kwargs)
    model_kwargs.update(cfg.model.hparams)

    # imputation loss
    loss_fn = torch_metrics.MaskedMAE()

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

    imputer = Imputer(model_class=model_cls,
                      model_kwargs=model_kwargs,
                      optim_class=getattr(torch.optim, cfg.optimizer.name),
                      optim_kwargs=cfg.optimizer.hparams,
                      loss_fn=loss_fn,
                      metrics=log_metrics,
                      impute_only_missing=False,
                      scheduler_class=scheduler_class,
                      scheduler_kwargs=scheduler_kwargs,
                      scale_target=False if scaler_cfg is None else scaler_cfg.scale_target,
                      whiten_prob=cfg.whiten_prob,
                      warm_up_steps=cfg.imputation_warm_up)

    # logger
    run_args = exp.get_config_dict()
    run_args['model']['trainable_parameters'] = imputer.trainable_parameters
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

    # Training

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

    trainer.fit(imputer,
                train_dataloaders=dm.train_dataloader(),
                val_dataloaders=dm.val_dataloader())

    # testing

    imputer.load_model(checkpoint_callback.best_model_path)

    imputer.freeze()
    trainer.test(imputer, datamodule=dm)

    # saving dataset predictions

    # creating the dataloader that contain the whole dataset
    dm.testset = np.arange(0, len(dm.torch_dataset))
    dm.splitter = None
    imputer.eval()

    # Predict on the test dataset without updating weights
    with torch.no_grad():
        y_hat_raw = trainer.predict(imputer, dataloaders=dm.test_dataloader())
        y_hat_tensors = [entry['y_hat'] for entry in y_hat_raw]
        combined_tensor = torch.cat(y_hat_tensors, dim=0).squeeze(-1).detach().cpu().numpy()  # Combine and remove last dimension if singular

    # Prediction dataframe aggregation
    index = dm.torch_dataset.data_timestamps(dm.testset.indices)['window']
    # Flatten windows
    index = pd.DatetimeIndex(index.reshape(-1))
    combined_tensor = combined_tensor.reshape(-1, *combined_tensor.shape[2:])
    # df_hats = dict(zip(aggr_methods, df_hats))

    # create the directory to dinamically save the imputation
    directory_path = os.path.join('/home/smugnai/Thesis_Imputation', cfg.dir_imp)
    os.makedirs(directory_path, exist_ok=True)
    df = pd.DataFrame(data=combined_tensor, index=index,
                      columns=dataset._columns_multiindex())

    time_set = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = os.path.join(directory_path, f'imputed_dataset_with_timestamps_{time_set}.h5')
    df.to_hdf(file_path, key='imputed_dataset', mode='w', complevel=3)


if __name__ == '__main__':
    suppress_known_warnings()
    exp = Experiment(run_fn=run,
                     config_path='../config/',
                     config_name='default_imputation')
    res = exp.run()
    logger.info(res)
