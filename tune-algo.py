"""
Tune hyperparameters for an algorithm.

Usage:
    tune-algo.py [options] <dir> <algo>

Options:
    -v, --verbose
        Increase logging verbosity.
    -r FILE, --record=FILE
        Record individual points to FILE.
    -o FILE
        Save parameters to FILE.
"""

import sys
from pathlib import Path
import logging
from importlib import import_module
import json
import csv

from docopt import docopt
import pandas as pd
import numpy as np

from lenskit import batch, topn
from lenskit.algorithms import Recommender
import seedbank

_log = logging.getLogger('tune-algo')

def sample(space, state):
    "Sample a single point from a search space."
    return {
        name: dist.rvs(random_state=state)
        for (name, dist) in space
    }


def evaluate(point):
    "Evaluate the algorithm with a set of parameters."
    algo = algo_mod.from_params(**point)
    _log.info('evaluating %s', algo)

    algo = Recommender.adapt(algo)
    algo.fit(train_data)

    recs = batch.recommend(algo, test_users, 5000)
    rla = topn.RecListAnalysis()
    rla.add_metric(topn.recip_rank, k=5000)
    scores = rla.compute(recs, test_data, include_missing=True)
    mrr = scores['recip_rank'].fillna(0).mean()
    return mrr

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

    state = seedbank.numpy_random_state()

    points = []
    record_fn = args['--record']
    if record_fn:
        rcols = [name for (name, _dist) in algo_mod.space]
        rcols.append('mrr')
        record = csv.DictWriter(open(record_fn, 'w'), rcols)
        record.writeheader()
    else:
        record = None

    for i in range(60):
        point = sample(algo_mod.space, state)
        _log.info('iter %d: %s', i + 1, point)
        mrr = evaluate(point)
        _log.info('iter %d: MRR=%0.4f', i + 1, mrr)
        point['mrr'] = mrr
        points.append(point)
        if record:
            record.writerow(point)

    points = sorted(points, key=lambda p: p['mrr'], reverse=True)
    best_point = points[0]
    _log.info('finished in with MRR %.3f', best_point['mrr'])
    for p, v in best_point.items():
        _log.info('best %s: %s', p, v)

    fn = args.get('-o', None)
    if fn:
        _log.info('saving params to %s', fn)
        Path(fn).write_text(json.dumps(best_point))


if __name__ == '__main__':
    args = docopt(__doc__)
    main(args)
