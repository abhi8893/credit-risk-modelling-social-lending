import re
import functools
import numpy as np
import pandas as pd

def na_patch(func):
    @functools.wraps(func)
    def modfunc(*args, **kwargs):

        try:
            val_to_check = args[0]
        except IndexError:
            val_to_check = list(kwargs.values())[0]

        if pd.isna(val_to_check):
            return np.nan

        return func(*args, **kwargs)

    return modfunc

@na_patch
def remove_extra_whitespace(s):
    '''remove extra whitespace from string'''
    
    return re.sub(r'\s+', r' ', s.strip())

def remove_high_miss_columns(df, thresh):
    miss_df = df.isna().mean()
    low_miss_cols = miss_df[(miss_df < thresh)].index
    return df[low_miss_cols]

def get_string_numeric_stats(df):
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    str_cols = df.select_dtypes(include='object').columns

    str_stats = df[str_cols].apply(lambda col: tuple(col.sort_values().unique()))
    numeric_stats = df[numeric_cols].apply(lambda col: [col.min(), col.max()])
    numeric_stats = numeric_stats.T.apply(tuple, axis=1)

    return str_stats, numeric_stats


