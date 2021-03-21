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