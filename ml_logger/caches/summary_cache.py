from collections import defaultdict, deque

import more_itertools as mit
import numpy as np

AVERAGE_MODES = ["rolling", 'tiled']
STAT_MODES = ["mean", "min_max", "std_dev", "quantile", "histogram"]


class SummaryCache:
    def __init__(self, mode="tiled", default_stats="mean", window: int = None):
        """

        :param mode:
        :param window: the rolling average window under rolling mode
        """
        self.mode = mode
        if mode == "rolling":
            self.data = defaultdict(lambda: deque(maxlen=window))
        elif mode == "tiled":
            assert window is None, "shouldn't specify window under tiled mode"
            self.data = defaultdict(list)
        else:
            raise KeyError(f'mode `{mode}` is not supported. Allowed modes: {AVERAGE_MODES}.')

        self.default_stats = default_stats

    def __bool__(self):
        """returns True if any of the values is non-empty."""
        for v in self.data.values():
            if bool(v):
                return True
        return False

    def __repr__(self):
        s = "SummaryCache:"
        for k, v in self.data.items():
            try:
                s += f"  {k}: Shape{v[0].shape}[:{len(v)}]\n"
            except:
                s += f"  {k}: {v[:2]}[:{len(v)}]\n"
        return s

    def __contains__(self, item):
        return item in self.data

    def store(self, metrics=None, **key_values):
        """
        Store the metric data for making the summary later. This allows the logging/saving
        of training metrics asynchronously from the logging.

        :param metrics: (mapping) a mapping of metrics key/values. Will be destructured and appended
                        to the data store one key/value at a time,
        :param ** key_values: (Any) key/value arguments, each being a metric key / metric value pair.
        :return: None
        """
        keys = []
        if metrics:
            key_values.update(metrics)
        for k, v in key_values.items():
            self.data[k].append(v)
            keys.append(k)
        return keys

    def summarize(self, force_clear=None, key_stats=None, **_key_stats):
        """
        | summarizes the statistics, and clears the data store if
        |     1. `force_clear` is set to True
        |       OR
        |     2. the cache is under `tiled` mode.
        |
        | Note that this method is NOT Idempotent. Clears the data store if not using rolling average.
        | Because of this, summarize does not support explicit mode of `get_stats`. The summary
        | of all metric fields are returns when calling this method.
        |
        :param force_clear: (bool) forces clear the data store
        :param key_stats: (dict) a dictionary for the key and the statistic modes to be returned.
        :param _key_stats: (**) key value pairs, as a shorthand for the key_modes dictionary.
        :return: dictionary of the keys and the statistics requested.
        """
        summary = self.summary_stats(key_stats=key_stats, **_key_stats)
        if force_clear or self.mode == "tiled":
            self.clear()
        return summary

    def clear(self):
        self.data = self.data.copy()
        self.data.clear()

    def peek(self, *keys, len=5):
        """
        returns truncated version of the cached metrics, useful for taking a peek of
        what is saved in the dataset at the moment.

        :param keys:
        :param len:
        :return:
        """
        return {k: self.data[k][-len if len else None:] for k in (keys or self.data.keys()) if self.data[k] != []}

    def __getitem__(self, item):
        return self.get(item)

    def get(self, key, default=None):
        """
        Return the summary statistics or default

        :param key: the key for the metric
        :param default: None
        :return:
        """
        if key in self.data:
            cached = self.data[key]
            return np.array(cached)
        else:
            return default

    def pop(self, key, default=None):
        """
        Return the summary staatistics or default, and remove the key from the dictionary.

        example:
            avg_loss, = summary_cache.get('loss', stats='mean')

        :param key:
        :param default:
        :return:
        """
        if key in self.data:
            cached = self.data.pop(key)
            return np.array(cached)
        else:
            return default

    # note: is idempotent
    def summary_stats(self, *only_keys, key_stats=None, explicit=None, stats=None, **_key_stats):
        """
        Returns the metrics as a dictionary containing key / value pairs
        of the statistics of the metrics.

        OR

        Returns statistics that are queried.

        Note that this method is idempotent. AKA you can call multiple times without
        affecting what is stored in the data object.
        
        When a few key strings are passed in explicitly, ONLY those keys
        plus those specified in the keyword arguments in **key_modes are
        returned.
        
        To enable explicit mode without specifying *only_keys, set
        `get_only` to True
        
        # Modes for the Statistics:
        ===================
        
        key_mode would be one of:
          - mean:
          - min_max:
          - std_dev:
          - quantile:
          - histogram(bins=10):
        
        :param * only_keys: list of key strings to return explicitly
        :param key_stats: a dictionary for the key and the statistic modes to be returned.
        :param explicit: boolean flag, when true only keys explicitly specified would be returned
        :param stats: the default stats mode for all metrics
        :param ** _key_stats: key value pairs, as a short hand for the key_modes dictionary.
        :return: dictionary of the keys and the statistics requested
        # :return: dictionary of the keys and the statistics requested, or the value itself if only
        #         one value is specified.
        """
        explicit = explicit or len(only_keys) > 0
        key_stats = key_stats or dict()
        key_stats.update(_key_stats)
        metrics = dict()
        keys = {*only_keys, *key_stats.keys()} if explicit else self.data.keys()
        for k in keys:
            stats_type = key_stats.get(k, stats or self.default_stats)
            if k not in self.data:
                continue
            else:
                # this converts the None to np.nan that can be properly handled.
                d = np.array(flatten(self.data[k]))
                d = d[~np.isnan(d)]
                if len(d) == 0:
                    continue
                if "sum" in stats_type:
                    metrics[k + "/sum"] = d.sum()
                if "max" in stats_type:
                    metrics[k + "/max"] = d.max()
                if "min" in stats_type:
                    metrics[k + "/min"] = d.min()
                if "mean" in stats_type:
                    metrics[k + "/mean"] = d.mean()
                if "min_max" == stats_type:
                    metrics[k + "/mean"] = d.mean()
                if "std" in stats_type:
                    metrics[k + "/stddev"] = d.std()
                    metrics[k + "/mean"] = d.mean()
                    mode = mit.first(stats.mode(d, axis=None)[0], np.nan)
                    metrics[k + "/mode"] = mode
                if "quantile" in stats_type:
                    metrics[k + "/0"] = np.percentile(d, 0)
                    metrics[k + "/25"] = np.percentile(d, 25)
                    metrics[k + "/mean"] = np.percentile(d, 50)
                    metrics[k + "/75"] = np.percentile(d, 75)
                    metrics[k + "/100"] = np.percentile(d, 100)
                if "histograph" in stats_type:
                    # note: need make bin number configurable
                    metrics[k + "/hist"], metrics[k + "/divs"] = np.histogram(d, bins=10)

        return {k: float(v) for k, v in metrics.items()}


def to_float(item):
    if item is None:
        return float('nan')
    return float(item)


def flatten(array):
    """
    Returns a flattened nested arrays. Works with numpy containers with non-equal-length children.

    we don't use np.flatten, b/c sometimes you have to iterate through each child.

    Cast None to float('nan') values.

    :param array:
    :return:
    """
    r = []
    try:
        for item in array:
            r += flatten(item)
    except TypeError:  # Iterating through a 0-d array
        r += [to_float(array)]
    return r
