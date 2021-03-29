import numpy as np
import pandas as pd


def get_exclude_column_names(days):
    exclude_column_names=[]
    for i in days:
        exclude_column_names.append('next_'+str(i)+'_high')
    return exclude_column_names


def rename_reference_df_column_names(df, suffix):
    """
    example usage
        df_dji = rename_reference_df_column_names(df_dji, "_dji")
    ['c', 'h', 'l', 'o', 's', 't', 'v', 'wma'] -> ['c_dji', 'h_dji', 'l_dji', 'o_dji', 's_dji', 't_dji', 'v_dji', 'wma_dji']
    :param df: df to operate
    :param suffix: suffix to add to append
    :return:
    """
    old_names=df.columns
    # print(old_names)
    new_names=[s + suffix for s in old_names]
    # print(new_names)
    df.columns = new_names
    return df


def merge_dfs(df_left, df_right, col, suffix):
    """
    merge two df based on col & suffix
    :param df_left: left df, original stock like `GME`
    :param df_right: right df, some indexes like `^DJI`
    :param col: column to join two dfs
    :param suffix: in this case we've set it to _dji or _(ticker)
    :return:
    """
    # print(pd.merge(left=df_left, right=df_right, left_on='t', right_on='t' + suffix).head())
    df_merged = pd.merge(left=df_left, right=df_right, left_on=col, right_on=col + suffix)
    return df_merged


def generate_time_fields(df_dji):
    df_dji['ts'] = pd.to_datetime(df_dji['t'], unit='s')
    df_dji['date'] = pd.to_datetime(df_dji['t'], unit='s').dt.date


def get_numerical_df(df):
    return df.select_dtypes(include=np.number).reindex()