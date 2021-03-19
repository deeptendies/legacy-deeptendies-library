def get_exclude_column_names(days):
    exclude_column_names=[]
    for i in days:
        exclude_column_names.append('next_'+str(i)+'_high')
    return exclude_column_names