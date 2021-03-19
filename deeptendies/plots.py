from matplotlib import pyplot as plt


def get_line_plot(df, title = "Price vs. Date", x_step = 5):
  """Gets line plot for a standard finnhub df
  Params:
    df: df to plot
    title: name of plot
    x_step: number of time steps to print on x axis (ie., x_steps per tick). Note that buisness days only ploted!
  Returns:
    plt.fig instance
  """
  fig, ax = plt.subplots(figsize=(24,18))
  fig = plt.plot(range(df.shape[0]),(df['l']+df['h'])/2.0)
  plt.xticks(range(0,df.shape[0],x_step),df['t'].loc[::x_step],rotation=45)
  plt.xlabel('Date',fontsize=18)
  plt.ylabel('Mid Price',fontsize=18)
  plt.title(title)
  return fig


def plt_visual_raw(stock_sym, metric_interested, df):
    plt.figure(figsize=(16, 8))
    plt.title(stock_sym + " " + metric_interested + ' Price History')
    plt.plot(df[metric_interested])
    plt.xlabel('Date', fontsize=18)
    plt.ylabel(stock_sym + " " + metric_interested + ' Price USD ($)', fontsize=18)
    plt.show()


def plot_predicted(metric_interested, train, valid):
    # Visualize the data
    plt.figure(figsize=(16, 8))
    plt.title(metric_interested + ' Model')
    plt.xlabel('Date', fontsize=18)
    plt.ylabel(metric_interested + ' Price USD ($)', fontsize=18)
    plt.plot(train[metric_interested])
    plt.plot(valid[[metric_interested, 'Predictions']])
    plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
    plt.show()