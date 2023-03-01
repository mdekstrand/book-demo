"""
Split data for evaluation.

Usage:
    split-data.py [options] <input> <outdir>

Options:
    -v, --verbose
        Increase logging verbosity.
"""

import sys
from pathlib import Path
import logging

from docopt import docopt
import pandas as pd
import lenskit.crossfold as xf
import seedbank

_log = logging.getLogger('split-data')


def main(args):
    level = logging.DEBUG if args['--verbose'] else logging.INFO
    logging.basicConfig(level=level, stream=sys.stderr)
    logging.getLogger('numba').setLevel(logging.INFO)

    seedbank.initialize(20230301)

    infile = args['<input>']
    outdir = Path(args['<outdir>'])
    outdir.mkdir(exist_ok=True, parents=True)
    _log.info('loading ratings from %s', infile)
    ratings = pd.read_parquet(infile)
    _log.info('%d ratings from %d users for %d items',
              len(ratings), ratings['user'].nunique(), ratings['item'].nunique())

    _log.info('creating two test sets')
    splits = xf.sample_users(ratings, 2, 2500, xf.SampleN(5))
    for name, split in zip(['tune', 'eval'], splits):
        train, test = split
        train_fn = outdir / f'{name}-train.parquet'
        _log.info('saving %s', train_fn)
        train.to_parquet(train_fn, compression='zstd')
        test_fn = outdir / f'{name}-test.parquet'
        _log.info('saving %s', test_fn)
        test.to_parquet(test_fn, compression='zstd')


if __name__ == '__main__':
    args = docopt(__doc__)
    main(args)
