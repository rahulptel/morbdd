import numpy as np
from operator import itemgetter


def get_order(order_type, data):
    if order_type == "MinWt":
        idx_weight = [(i, w) for i, w in enumerate(data["weight"])]
        idx_weight.sort(key=itemgetter(1))

        return np.array([i[0] for i in idx_weight])
    elif order_type == "MaxRatio":
        min_profit = np.min(data["value"], 0)
        profit_by_weight = [v / w for v, w in zip(min_profit, data["weight"])]
        idx_profit_by_weight = [(i, f) for i, f in enumerate(profit_by_weight)]
        idx_profit_by_weight.sort(key=itemgetter(1), reverse=True)

        return np.array([i[0] for i in idx_profit_by_weight])
    elif order_type == "Lex":
        return np.arange(data["n_vars"])
