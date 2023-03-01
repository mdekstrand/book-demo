"""
Tune hyperparameters for an algorithm.

Usage:
    tune-algo.py [options] <dir> <algo>

Options:
    -v, --verbose
        Increase logging verbosity.
    -o FILE
        Save parameters to FILE.
"""

import sys
from pathlib import Path
import logging
from importlib import import_module
import json

from docopt import docopt
import pandas as pd
import numpy as np
from skopt import gp_minimize
from scipy.optimize import OptimizeResult

from lenskit import batch, topn
from lenskit.algorithms import Recommender
import seedbank

_log = logging.getLogger('tune-algo')
niters = 0


def evaluate(space):
    "Evaluate the algorithm with a set of parameters."
    global niters
    niters += 1
    _log.info('iter %d: %s', space)
    algo = algo_mod.from_space(space)
    _log.info('evaluating %s', algo)

    algo = Recommender.adapt(algo)
    algo.fit(train_data)

    recs = batch.recommend(algo, test_users, 5000)
    rla = topn.RecListAnalysis()
    rla.add_metric(topn.recip_rank, k=5000)
    scores = rla.compute(recs, test_data, include_missing=True)
    mrr = scores['recip_rank'].fillna(0).mean()
    _log.info('iter %d: MRR=%0.4f', niters, mrr)
    return -mrr


def check_stop(result: OptimizeResult):
    if len(result.func_vals) >= 20:
        fvs = np.sort(result.func_vals)
        fvr = fvs[-5] - fvs[-1]
        return fvr <= 1.0e-3
    else:
        return False


def main(args):
    global algo_mod, train_data, test_data, test_users
    level = logging.DEBUG if args['--verbose'] else logging.INFO
    logging.basicConfig(level=level, stream=sys.stderr)
    logging.getLogger('numba').setLevel(logging.INFO)

    seedbank.initialize(20230301)

    algo_name = args['<algo>']
    _log.info('loading algorithm %s', algo_name)
    algo_mod = import_module(f'bookdemo.algorithms.{algo_name}')

    data = Path(args['<dir>'])
    _log.info('loading data from %s', data)
    train_data = pd.read_parquet(data / 'tune-train.parquet')
    test_data = pd.read_parquet(data / 'tune-test.parquet')
    test_users = test_data['user'].unique()

    res = gp_minimize(evaluate, algo_mod.space, random_state=seedbank.numpy_random_state(), callback=check_stop)
    _log.info('finished in %d steps with MRR %.3f', len(res.func_vals), res.fun)

    params = {}
    for dim, x in zip(algo_mod.space, res.x):
        _log.info('best value for %s: %s', dim, x)
        params[dim] = x

    fn = args.get('-o', None)
    if fn:
        _log.info('saving params to %s', fn)
        Path(fn).write_text(json.dumps(params))


if __name__ == '__main__':
    args = docopt(__doc__)
    main(args)
