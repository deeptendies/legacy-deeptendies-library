import os
import pathlib
from datetime import date

def save_data(dataframe, bucket, topic, version, suffix, path=None):
    """
    this helper function preps data emulating buckets key based structures.
    # usage example:
        from local_bucket import save_data
        accounting = [[10,'2020-04-08','buy',150,3.200000047683716,3.4460033373533414,600,998080.241169779,1000000.2411983892],
                      [11,'2020-04-09','sell',150,4.25,4.042784088016275,450,998676.6587829815,1000589.1587829815]]
        df = pd.DataFrame(accounting, columns=["day", "action price", "investment", "net worth"])
        print(df.describe())
        ticker = 'GME'
        start_date = '2020-04-01'
        end_date = '2021-04-01'
        save_data(dataframe=df, bucket='filesys', topic=ticker, version='actioned_reinforcement_learning',
        suffix=f"{start_date}_to_{end_date}")
    :param dataframe:
    :param bucket:
    :param topic:
    :param version:
    :param suffix:
    :param path: default saving to the same working dir, if specified will be saved to designated loc
    :return:
    """
    def df_to_filesys_operator(df, path, filename):
        if not os.path.exists(path):
            pathlib.Path(path).mkdir(parents=True, exist_ok=True)
        df.to_csv(os.path.join(path, filename))
        return

    df = dataframe
    bucket = 'bucket=' + bucket
    if path is None:
        path = bucket
    else:
        path = os.path.join(path, bucket)
    filename = topic + "_" + suffix + '.csv'
    topic = 'topic=' + topic
    processed_at = 'processed_at=' + date.today().strftime("%Y-%m-%d")
    version = 'version=' + version
    path = os.path.join(
        path,
        topic,
        version,
        processed_at
    )
    df_to_filesys_operator(df=df,
                           path=path,
                           filename=filename)
