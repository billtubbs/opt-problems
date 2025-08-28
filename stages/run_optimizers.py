import time
import sys
import os

import numpy as np
import pandas as pd
import scipy
import skopt
import yaml
import lpfgopt
from tqdm import tqdm

from problems.optprob.problems import solve_problem_with_optimizer
from problems.toy_1d import toy_1d_problem
from problems.sys_id_fopdt import sys_id_fopdt
from optimizers.optimizers import (
    scipy_minimize_rand_init,
    lpfgopt_minimize_rand_init
)


RESULTS_DIR = 'results'
STAGE_RESULTS_DIR = 'results'

# Register all problem classes
problems = {
    "Toy1DProblem": toy_1d_problem.Toy1DProblem,
    "SysIdFromFileFOPDT": sys_id_fopdt.SysIdFromFileFOPDT,
    "SysIdFromFileFOPDTRealDelay": sys_id_fopdt.SysIdFromFileFOPDTRealDelay,
    }

# Register all optimizer classes
optimizers = {
    "scipy_optimize_minimize": scipy.optimize.minimize,
    "lpfgopt_minimize": lpfgopt.minimize,
    "skopt_gp_minimize": skopt.gp_minimize,
    "scipy_minimize_rand_init": scipy_minimize_rand_init,
    "lpfgopt_minimize_rand_init": lpfgopt_minimize_rand_init,
}


def num_digits(n):
    """Return the number of digits needed to represent an integer.
    """
    return len(str(abs(n)))


def run_optimizers(exp_name, exp_params):
    """Run optimizer on problem n_trials times and save results to file.
    """

    # Instantiate optimization problem class
    problem_name = exp_params['problem']['name']
    args = list(exp_params['problem'].get('args', {}).values())
    kwargs = exp_params['problem'].get('kwargs', {})
    problem = problems[problem_name](*args, **kwargs)

    # Collect summary stats of all trials
    exp_stats = {}

    pbar_outer = tqdm(exp_params['optimizers'].items(), desc="Optimizer")
    for opt_name, opt_params in pbar_outer:
        pbar_outer.set_description(opt_name)

        # Prepare directory to save results
        os.makedirs(
            os.path.join(RESULTS_DIR, exp_name, STAGE_RESULTS_DIR, opt_name),
            exist_ok=True
        )

        # Call optimizer function
        opt_class_name = opt_params.get('name', opt_name)
        optimizer = optimizers[opt_class_name]
        args = list(opt_params.get('args', {}).values())
        kwargs = opt_params.get('kwargs', {})

        n_trials = opt_params.get('n_trials', 1)

        # Used to make sure filenames are correctly sortable
        nd = num_digits(n_trials)

        for trial in tqdm(range(n_trials), desc="Trials", leave=False):

            # Call optimizer
            t_start = time.time()
            res = solve_problem_with_optimizer(
                problem, optimizer, *args, **kwargs
            )
            elapsed_time = time.time() - t_start

            # Save optimizer function call history
            f_values, guesses = zip(*problem.guesses)
            f_values = np.array(f_values)
            guesses = np.stack(guesses)
            opt_history = pd.concat(
                [
                    pd.Series(f_values, name='f'),      
                    pd.DataFrame(
                        guesses,
                        columns=[
                            f'x{i+1}' for i in range(guesses.shape[1])
                        ]
                    ),
                ],
                axis=1
            )
            opt_history.index.name = 'iter'
            filename = f"fevals_{trial:0{nd}d}.csv"
            filepath = os.path.join(
                RESULTS_DIR, exp_name, STAGE_RESULTS_DIR, opt_name, filename
            )
            opt_history.to_csv(filepath)

            # Save best guess and summary stats
            f, x = problem.best_guess
            stats = {
                'nfev': problem.nfev,
                'elapsed_time': elapsed_time,
                'f': float(f),
            }
            x = np.array(x).tolist()
            stats.update({f'x{i}': xi for i, xi in enumerate(x)})
            optional_attributes = ['success', 'status']
            for name in optional_attributes:
                if name in res:
                    stats[f'res_{name}'] = getattr(res, name)
            exp_stats[(problem_name, opt_name, trial)] = stats

    exp_stats = pd.DataFrame.from_dict(exp_stats, orient='index')
    exp_stats.index.names = ["problem", "optimizer", "trial"]
    filepath = os.path.join(
        RESULTS_DIR, exp_name, STAGE_RESULTS_DIR, 'stats.csv'
    )
    exp_stats.to_csv(filepath)


def make_plots(exp_params):
    pass


if __name__ == "__main__":

    exp_name = sys.argv[1]
    with open("params.yaml", encoding="utf-8") as f:
        params = yaml.safe_load(f.read())
    exp_params = params['experiments'][exp_name]

    run_optimizers(exp_name, exp_params)
