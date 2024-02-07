from pathlib import Path
from functools import wraps

from sklearn_crfsuite.utils import flatten


def get_root_directory():
    return Path(__file__).parent.parent


def _flattens_y(func):
    @wraps(func)
    def wrapper(y_true, y_pred, *args, **kwargs):
        y_true_flat = flatten(y_true)
        y_pred_flat = flatten(y_pred)
        return func(y_true_flat, y_pred_flat, *args, **kwargs)

    return wrapper


@_flattens_y
def flat_classification_report(y_true, y_pred, labels=None, **kwargs):
    """
    Return classification report for sequence items.
    #NOTE: Adapated from source code to mitigate TypeError issue with original function.
    See: https://www.reddit.com/r/learnpython/comments/swwplz/sklearn_crfsuite_issue_using_metricsflat/
    """
    from sklearn import metrics

    return metrics.classification_report(y_true, y_pred, labels=labels, **kwargs)
