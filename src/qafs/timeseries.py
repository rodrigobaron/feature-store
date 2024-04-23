import functools

import pandas as pd


def concat(dfs):
    """Concat dataframes for multiple features."""
    return pd.concat(dfs, join="outer", axis=1).ffill()


def transform(df, func):
    """Transform dataframe using function."""
    transformed = func(df)
    if isinstance(df, pd.DataFrame) and not isinstance(transformed, pd.DataFrame):
        raise RuntimeError("Transforms in this namespace should return Pandas dataframes or series")
    if len(transformed.columns) != 1:
        raise RuntimeError(
            "Transform function should return a dataframe with a datetime index and single value column"
        )
    return transformed