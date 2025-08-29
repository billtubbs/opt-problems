
import os
import sys
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from problems.optprob.plot_utils import (
    best_guesses_plot, 
    best_guesses_plot_n_repeats
)


RESULTS_DIR = 'results'
PREV_STAGE_RESULTS_DIR = 'results'
STAGE_RESULTS_DIR = 'plots'


def make_f_histogram(
    stats, exp_name, opt_name, n_bins=51, ax=None, figsize=(5, 2)
):
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    else:
        ax = plt.gca()
    f_range = (stats['f'].min(), stats['f'].max())
    bin_width = (f_range[1] - f_range[0]) / (n_bins - 1)
    # bin_centres = np.linspace(f_range[0], f_range[1], n_bins - 1)
    bin_edges = np.linspace(
        f_range[0] - bin_width / 2,
        f_range[1] + bin_width / 2,
        n_bins
    )
    stats['f'].plot.hist(
        ax=ax,
        bins=bin_edges,
        xlabel='f(x)',
        grid=True,
        title=f'Solutions - {exp_name} - {opt_name}',
    )
    return ax


def make_plots(exp_name, exp_params, colors=None):
    """Make plots of results and save as image files."""

    # Load summary stats file
    filepath = os.path.join(
        RESULTS_DIR,
        exp_name,
        PREV_STAGE_RESULTS_DIR,
        "stats.csv"
    )
    stats = pd.read_csv(filepath)

    if colors is None:
        colors = [f"C{i}" for i in range(10)]

    fig_main, ax_main = plt.subplots(figsize=(7, 2.5))

    for color, opt_name in tqdm(zip(colors, exp_params['optimizers'])):

        # Prepare directory to save plot images
        os.makedirs(
            os.path.join(RESULTS_DIR, exp_name, STAGE_RESULTS_DIR, opt_name),
            exist_ok=True
        )

        # Make histogram of best f(x) values for all trials
        fig, ax = plt.subplots()
        make_f_histogram(
            stats.loc[stats['optimizer'] == opt_name],
            exp_name,
            opt_name,
            ax=ax,
        )
        fig.tight_layout()
        filename = f"f_histogram_{exp_name}.pdf"
        filepath = os.path.join(
            RESULTS_DIR, exp_name, STAGE_RESULTS_DIR, opt_name, filename
        )
        fig.savefig(filepath)
        plt.close(fig)

        # Make time-series plot of best guess vs. number of f(x) evaluations
        prev_stage_results_dir = os.path.join(
            RESULTS_DIR, exp_name, PREV_STAGE_RESULTS_DIR, opt_name
        )
        filenames = sorted(
            name for name in os.listdir(prev_stage_results_dir)
            if name.startswith('fevals_')
        )
        fun_evals = {}
        x_values = {}
        x_names = None
        for i, filename in enumerate(filenames):
            data = pd.read_csv(
                os.path.join(prev_stage_results_dir, filename),
                index_col=0
            )
            if x_names is None:
                x_names = [name for name in data.columns if name.startswith('x')]
            else:
                # Check the x variable names are identical
                assert (
                    set([name for name in data.columns if name.startswith('x')])
                    == set(x_names)
                )
            fun_evals[i+1] = data['f']
            x_values[i+1] = data[x_names]
        fun_evals = pd.concat(fun_evals, axis=1)
        x_values = pd.concat(x_values, axis=1)
        n_trials = fun_evals.shape[1]
        title = f'Convergence - {exp_name} - {opt_name} - {n_trials} Trials'

        fig, ax = plt.subplots(figsize=(7, 2.5))
        best_guesses_plot_n_repeats(
            fun_evals.to_numpy().T, ax=ax, color=color, title=title
        )
        fig.tight_layout()
        filename = f"best_guesses_tsplot_{exp_name}.pdf"
        filepath = os.path.join(
            RESULTS_DIR, exp_name, STAGE_RESULTS_DIR, opt_name, filename
        )
        fig.savefig(filepath)
        plt.close(fig)

        # Add same plot lines to main plot
        best_guesses_plot_n_repeats(
            fun_evals.to_numpy().T,
            ax=ax_main,
            color=color,
            label=opt_name,
            area_label=''
        )

    # TODO: Make this a parameter
    ax_main.set_xlim([0, 1000])
    fig_main.tight_layout()
    filename = "best_guesses_tsplot_all.pdf"
    filepath = os.path.join(
        RESULTS_DIR, exp_name, STAGE_RESULTS_DIR, filename
    )
    fig_main.savefig(filepath)
    plt.close(fig_main)


if __name__ == "__main__":

    exp_name = sys.argv[1]
    with open("params.yaml", encoding="utf-8") as f:
        params = yaml.safe_load(f.read())
    exp_params = params['experiments'][exp_name]

    make_plots(exp_name, exp_params)
