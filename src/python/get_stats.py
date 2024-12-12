import numpy as np


def get_stats(v):
    return [
        {
            "min": np.min(v),
            "max": np.max(v),
            "mean": np.mean(v),
            "sum": np.sum(v),
            "std": np.std(v),
            "var": np.var(v),
        },
        np.percentile(v, q=[10, 25, 50, 75, 90, 99]),
    ]
