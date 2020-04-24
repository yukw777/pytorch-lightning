from functools import wraps
import warnings


def rank_zero_only(fn):

    @wraps(fn)
    def wrapped_fn(*args, **kwargs):
        if rank_zero_only.rank == 0:
            return fn(*args, **kwargs)

    return wrapped_fn


def _warn(*args, **kwargs):
    warnings.warn(*args, **kwargs)


rank_zero_only.rank = getattr(rank_zero_only, 'rank', 0)
rank_zero_warn = rank_zero_only(_warn)
