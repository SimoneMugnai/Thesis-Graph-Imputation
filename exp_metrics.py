import os
from argparse import ArgumentParser

import neptune.new as neptune
import numpy as np
import pandas as pd

IGNORE_PARAMS = ['tags', 'run/*', 'task']
VERBOSE = True
SEED_KEY = 'run/seed'


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--project', type=str, default='user/project')
    parser.add_argument('--project-key', type=str, default='PROJ')
    parser.add_argument('--owner', type=str, default=None)
    parser.add_argument('--runs', nargs='+', type=int, default=None)
    parser.add_argument('--tags', nargs='+', type=str, default=None)
    parser.add_argument('--models', nargs='+', type=str, default=None)
    parser.add_argument('--datasets', nargs='+', type=str, default=None)
    parser.add_argument('--absolute-metrics', nargs='+', type=str,
                        default=['test_mae'])
    parser.add_argument('--relative-metrics', nargs='+', type=str,
                        default=[])
    parser.add_argument('--absolute-decimals', type=int, default=2)
    parser.add_argument('--relative-decimals', type=int, default=2)
    parser.add_argument('--omit-zero', action='store_true', default=False)
    parser.add_argument('--omit-std', action='store_true', default=False)
    parser.add_argument('--bold-best', action='store_true', default=False)
    parser.add_argument('--trim-spaces', action='store_true', default=False)
    parser.add_argument('--silent', action='store_true', default=False)
    parser.add_argument('--heatmap-color', type=str, default=None)
    parser.add_argument('--heatmap-min', type=float, default=0.)
    parser.add_argument('--heatmap-max', type=float, default=1.)
    parser.add_argument('--heatmap-intensity-max', type=int, default=None)
    parser.add_argument('--ignore-params', nargs='+', type=str, default=[])
    args = parser.parse_args()

    global VERBOSE
    VERBOSE = not args.silent
    if not VERBOSE:
        os.environ['NEPTUNE_MODE'] = 'debug'

    if args.tags is not None:
        log(f'Tags: {list(args.tags)}')
    if args.runs is not None:
        args.runs = [f'{args.project_key}-{run}' for run in args.runs]
        log(f'Runs: {list(args.runs)}')
    args.ignore_params = IGNORE_PARAMS + args.ignore_params
    args.metrics = args.absolute_metrics + args.relative_metrics
    return args


def log(text: str):
    if VERBOSE:
        print(text)


def custom_filter(logs: pd.DataFrame, parameters: pd.DataFrame):
    # Implement here your custom filtering logic
    return logs, parameters


def fetch_runs_table(project_name, state=None, owner=None, runs=None,
                     tags=None, filter_trashed=True):
    assert state in [None, 'idle', 'running']
    project = neptune.init_project(project_name, mode='read-only')
    run_table_df = project.fetch_runs_table(state=state, owner=owner,
                                            tag=tags).to_pandas()
    if not len(run_table_df):
        return None

    if filter_trashed:
        run_table_df = run_table_df[~ run_table_df['sys/trashed']]

    # set run id as index
    run_table_df = run_table_df.set_index('sys/id', drop=True).sort_index()

    if runs is not None:
        run_table_df = run_table_df.loc[runs]

    dfs = {'logs': None, 'parameters': None}
    for key in dfs:
        # filter cols
        cols = {col: col[len(key) + 1:] for col in run_table_df.columns if
                col.startswith(key)}
        dfs[key] = run_table_df.loc[:, cols.keys()].rename(
            columns=cols).convert_dtypes()
        # cast values
        for dtype in ['int', 'float']:
            try:
                cols_dtype = dfs[key].select_dtypes(f'{dtype}64').columns
                dfs[key].loc[:, cols_dtype] = dfs[key].loc[:,
                                              cols_dtype].astype(f'{dtype}32')
            except:
                pass
    return dfs


def check_params(parameters, ignore=None):
    if ignore is None:
        ignore = set()
    n_exps = len(parameters)

    #  Check if all experiments have different seed  ##########################
    if SEED_KEY is not None:
        assert len(set(parameters.loc[:, SEED_KEY])) == n_exps, \
            "Not all the experiments have different seeds!"

    #  Remove columns with all NaNs  ##########################################
    parameters = parameters.loc[:, ~(parameters.isna().all())]

    #  Check if all experiments have same parameters  #########################
    params = set(parameters.columns)
    ignore_params = set() if SEED_KEY is None else {SEED_KEY}
    # select parameters to be checked
    for del_param in ignore:
        if del_param in params:
            ignore_params.add(del_param)
        elif del_param.endswith('*'):
            ignore_params.update({p for p in params
                                  if p.startswith(del_param[:-1])})
    # check all shared parameters
    for col in params.difference(ignore_params):
        assert len(set(parameters.loc[:, col])) == 1, \
            f"Parameter {col} is not the same in all the experiments"

    log('All the experiments have the same parameters and different seed.')


def cell_color(color: str, value: float, range_min: float, range_max: float,
               max_intensity: int = None):
    intensity = int((value - range_min) / (range_max - range_min) * 100)
    if max_intensity is not None:
        intensity = min(intensity, max_intensity)
    if intensity <= 0:
        return ""
    color_txt = f"{color}!{intensity}"
    return "\cellcolor{" + color_txt + "}"


def make_experiment_table(logs, parameters, args):
    #  Group runs by dataset and model  #######################################
    runs_mask = []
    if args.datasets is not None:
        runs_mask.append(parameters['dataset/name'].isin(args.datasets))
    if args.models is not None:
        runs_mask.append(parameters['model/name'].isin(args.models))
    if not len(runs_mask):
        raise ValueError('No dataset or model specified.')
    # Select runs that match all the conditions  ##############################
    runs_mask = pd.concat(runs_mask, axis=1).all(axis=1)
    logs = logs.loc[runs_mask]
    parameters = parameters.loc[runs_mask]
    n_exps = len(logs)
    log(f'{n_exps} experiments after filtering: {list(logs.index)}')
    # Group runs by dataset and model  ########################################
    groups = parameters.groupby([parameters['dataset/name'],
                                 parameters['model/name']]).groups
    #  Create dataframe to store metrics  #####################################
    filter_metrics = logs.columns[logs.columns.isin(args.metrics)]
    columns = pd.MultiIndex.from_product([args.datasets,
                                          filter_metrics,
                                          ['mean', 'std']],
                                         names=['dataset', 'metric', 'stat'])
    df = pd.DataFrame(index=args.models, columns=columns)
    #  Compute metrics statistics  ############################################
    for (dataset, model), idx in groups.items():
        #  Check runs' parameters (e.g., different seed, same config)  ########
        log(f'Group: {dataset} - {model} ({len(idx)} runs): {idx.values}')
        check_params(parameters.loc[idx], ignore=args.ignore_params)
        #  Compute metrics statistics  ########################################
        metrics = logs.loc[idx, filter_metrics]
        # scale relative metrics to %
        metrics.loc[:, args.relative_metrics] *= 100
        mean = metrics.mean()
        std = metrics.std()
        df.loc[model, (dataset, slice(None), 'mean')] = mean.values
        df.loc[model, (dataset, slice(None), 'std')] = std.values
    #  Format metrics in LaTeX table row  #####################################
    table = format_cell_latex(df, args)
    #  Print  #################################################################
    print("\n&", escape_str(" & ".join(table.columns.get_level_values(0))),
          end=" \\\\\n")
    print("&", escape_str(" & ".join(table.columns.get_level_values(1))),
          end=" \\\\\n")
    for model, values in table.iterrows():
        print(escape_str(model), end=" & ")
        print(" & ".join(values), end=" \\\\\n")
    print("\n")
    return table, df


def escape_str(txt):
    return txt.replace('_', '\\_')


def format_cell_latex(df, args):
    #  Format metrics in LaTeX table row  #####################################
    columns = df.columns.droplevel(2).unique()
    table = pd.DataFrame(index=df.index, columns=columns)
    for m in args.metrics:
        # format mean and std values, rounding up to specified decimals
        if m in args.absolute_metrics:
            decimals = args.absolute_decimals
        else:
            decimals = args.relative_decimals
        mu_val = df.loc[:, (slice(None), m, 'mean')].values.astype('float32')
        sigma_val = df.loc[:, (slice(None), m, 'std')].values.astype('float32')
        # Round up to specified decimals and convert to string
        mu = np.vectorize(lambda x: f"{x:.{decimals}f}")(mu_val)
        sigma = np.vectorize(lambda x: f"{x:.{decimals}f}")(sigma_val)
        # Extend strings to 64 chars
        mu, sigma = mu.astype('<U64'), sigma.astype('<U64')
        # Replace NaNs with --.---
        nans = np.isnan(mu_val)
        mu[nans], sigma[nans] = '--.---', '--.---'
        # if --omit-zero and values < 1, remove first zeros and plot .{decimals}
        if args.omit_zero:
            mu = np.char.replace(mu, '0.', '.')
            sigma = np.char.replace(sigma, '0.', '.')
        # if --omit-std, remove std deviation interval
        if args.omit_std:
            txt = mu
        else:
            txt = np.char.add(mu, '{{\\tiny $\\pm$')
            txt = np.char.add(txt, np.char.add(sigma, '}}')).astype('<U64')
        # add color cell if heatmap is specified
        if args.heatmap_color is not None:
            # todo
            pass
        # add color cell if heatmap is specified
        if args.bold_best:
            best = np.nanargmin(mu_val, axis=0)
            idx_ = (best, np.arange(mu_val.shape[1]))
            txt[idx_] = np.char.add('\\textbf{', txt[idx_])
            txt[idx_] = np.char.add(txt[idx_], '}')
        # if --trim-spaces, 0.468 ± 0.0 -> 0.468±0.0
        if args.trim_spaces:
            txt = np.char.replace(txt, ' ', '')
        table.loc[:, (slice(None), m)] = txt
    return table


def make_experiment_row(logs, parameters, args):
    #  Check runs' parameters (e.g., different seed, same config)  ############
    check_params(parameters, ignore=args.ignore_params)

    #  Compute metrics statistics  ############################################
    filter_metrics = [col for col in args.metrics if col in logs.columns]
    metrics = logs.loc[:, filter_metrics]
    # scale relative metrics to %
    metrics.loc[:, args.relative_metrics] *= 100
    mean = metrics.mean()
    std = metrics.std()

    #  Format metrics in LaTeX table row  #####################################
    mtr_values = dict()
    for m in metrics:
        # format mean and std values, rounding up to specified decimals
        if m in args.absolute_metrics:
            decimals = args.absolute_decimals
        else:
            decimals = args.relative_decimals
        mu, sigma = f"{mean[m]:.{decimals}f}", f"{std[m]:.{decimals}f}"
        # if --omit-zero and values < 1, remove first zeros and plot .{decimals}
        if args.omit_zero:
            mu = mu.replace('0.', '.')
            sigma = sigma.replace('0.', '.')
        # if --omit-std, remove std deviation interval
        if args.omit_std:
            txt = f"{mu}"
        else:
            txt = f"{mu} {{\\tiny $\\pm$ {sigma}}}"
        # add color cell if heatmap is specified
        if args.heatmap_color is not None:
            color = cell_color(args.heatmap_color, mean[m],
                               args.heatmap_min, args.heatmap_max,
                               max_intensity=args.heatmap_intensity_max)
            txt = color + txt
        # if --trim-spaces, 0.468 ± 0.0 -> 0.468±0.0
        if args.trim_spaces:
            txt = txt.replace(' ', '')
        mtr_values[m] = txt

    #  show metrics header for readability  ###################################
    lengths = {k: len(txt) - len(k) - 2 for k, txt in mtr_values.items()}
    header = []
    for key, length in lengths.items():
        pad = '=' * length
        half = length // 2
        header.append(pad[:half] + ' ' + key + ' ' + pad[half:])
    header = " | ".join(header)

    #  Print  #################################################################
    print(header)
    print(" & ".join(mtr_values.values()))


if __name__ == '__main__':
    args = parse_args()

    #  Get runs  ##############################################################
    exps = fetch_runs_table(project_name=args.project, owner=args.owner,
                            tags=args.tags, runs=args.runs, state=None)
    if exps is None:
        print('No experiments found.')
        exit()

    logs, parameters = exps['logs'], exps['parameters']
    # Optionally filter runs
    logs, parameters = custom_filter(logs, parameters)

    n_exps = len(logs)
    log(f'{n_exps} experiments found: {list(logs.index)}')

    #  Compute metrics statistics  ############################################
    if args.datasets is not None or args.models is not None:
        table, df = make_experiment_table(logs, parameters, args)
    else:
        make_experiment_row(logs, parameters, args)


### EXAMPLE USAGE ###
# python exp_metrics.py --absolute-metrics test_mae_unmasked --absolute-decimals 2 --trim-spaces --tags tab_imputation thesis --models rnn dcrnn agcrn gwnet tts_imp tts_amp --datasets la_point la_block --bold-best