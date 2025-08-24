import time
import sys
import os

import yaml
import lpfgopt
import scipy
import skopt
import numpy as np
import pandas as pd
from tqdm import tqdm
from problems.optprob.problems import solve_problem_with_optimizer
from problems.toy_1d import toy_1d_problem
from problems.sys_id_fopdt import sys_id_fopdt


# Register problems here
problems = {
    "Toy1DProblem": toy_1d_problem.Toy1DProblem,
    "SysIdFOPDT": sys_id_fopdt.SysIdFOPDT,
}

optimizers = {
    "scipy_optimize_minimize": scipy.optimize.minimize,
    "lpfgopt_minimize": lpfgopt.minimize,
    "skopt_gp_minimize": skopt.gp_minimize,
}

RESULTS_DIR = 'results'


if __name__ == "__main__":

    exp_name = sys.argv[1]

    with open("params.yaml", encoding="utf-8") as f:
        params = yaml.safe_load(f.read())

    for exp_name, exp_params in params['experiments'].items():

        os.makedirs(os.path.join(RESULTS_DIR, exp_name), exist_ok=True)

        # Instantiate optimization problem class
        problem_name = exp_params['problem']['name']
        args = exp_params['problem'].get('args', [])
        kwargs = exp_params['problem'].get('kwargs', {})
        problem = problems[problem_name](*args, **kwargs)
        
        exp_stats = {}
        for opt_name, opt_params in tqdm(exp_params['optimizers'].items()):

            # Call optimizer function
            opt_class_name = opt_params.get('name', opt_name)
            optimizer = optimizers[opt_class_name]
            args = opt_params.get('args', {}).values()
            kwargs = opt_params.get('kwargs', {})

            n_trials = opt_params.get('n_trials', 1)

            for trial in range(n_trials):

                # Call optimizer
                t_start = time.time()
                sol = solve_problem_with_optimizer(
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
                filename = f"{problem_name}_{opt_name}_fevals_{trial}.csv"
                opt_history.to_csv(os.path.join(RESULTS_DIR, exp_name, filename))

                # Save best guess and summary stats
                f, x = problem.best_guess
                stats = {
                    'nfev': problem.nfev,
                    'elapsed_time': elapsed_time,
                    'f': float(f),
                }
                x = np.array(x).tolist()
                stats.update({f'x{i}': xi for i, xi in enumerate(x)})
                exp_stats[(problem_name, opt_name, trial)] = stats

        exp_stats = pd.DataFrame.from_dict(exp_stats, orient='index')
        exp_stats.index.names = ["problem", "optimizer", "trial"]
        exp_stats.to_csv(os.path.join(RESULTS_DIR, exp_name, 'stats.csv'))
