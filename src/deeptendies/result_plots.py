import pandas as pd
from matplotlib import pyplot as plt
import plotly
from plotly.graph_objs import Scatter, Layout
import numpy as np
import plotly.graph_objects as go


class ModelResultPlots:
    """
    This class provides helper methods to
    reuse code needed for model result plots
    """
    @staticmethod
    def plot_predicted_vs_original_price(y_test, y_prediction, stock_name, days, x_step = 150):
            """
            Plots real vs. predicted stock price
            Params:
                y_test: test data array
                y_prediction: predicted prices array
                stock_name: stock name as a string
                days: list<string>: dates to plot 
                x_steps: Days to 
            Returns:
                displays plot
            """           
            fig, ax = plt.subplots(figsize=(24,18))
            ax.plot(days, y_test, label = 'Real ' + stock_name + ' stock price')
            ax.plot(days, y_prediction, label = 'Predicted ' + stock_name + ' stock price')
            plt.xticks(range(0,len(days),x_step), days[::x_step],rotation=45)
            plt.xlabel('Date',fontsize=24)
            plt.ylabel('Closing Price',fontsize=24)
            plt.title(f"{stock_name} Closing Price vs. Date", fontsize=36)
            plt.legend(loc='upper right')
            return fig 

    @staticmethod
    def plot_loss(history, train_metric_name, val_metric_name, title, ylabel, xlabel = "Epoch"):
        """
        Plots training and validation data vs epoch
        Params:
            history: model history object
            train_metric_name: string to access train data
            val_metric_name: string to access validation data
            title: plot title
            ylabel: plot's y-axis label
            xlabel: plot's x-axis label
        Returns:
            displays plot
        """
        train = history.history[train_metric_name]
        val = history.history[val_metric_name]
        fig, ax = plt.subplots(figsize=(18,12))
        ax.plot(range(len(train)), train, label=train_metric_name)
        ax.plot(range(len(train)), val, label=val_metric_name)
        ax.legend()
        fig.suptitle(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid(True)
        fig.show()

    @staticmethod
    def plot_metrics_values_predicted(metric_interested, train, valid):
        """
        Visualizes predicted values for specified metric
        Params:
            metric: the metric considered (e.g. accuracy)
            train: dataFrame
            valid: dataFrame
        Returns:
            displays plot
        """
        plt.figure(figsize=(16, 8))
        plt.title(metric_interested + ' Model')
        plt.xlabel('Date', fontsize=18)
        plt.ylabel(metric_interested + ' Price USD ($)', fontsize=18)
        plt.plot(train[metric_interested])
        plt.plot(valid[[metric_interested, 'Predictions']])
        plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
        plt.show()

    @staticmethod
    def plot_training_and_test_prediction(date_train, close_train, date_val, close_val, date_test, close_test, prediction, title):
        """
        Plots training, validation, test prediction, and actual test data
        Params:
            date_train: training x values
            date_val: validation x values
            date_test: test x values
            close_train: closing price training data
            close_val: closing price validation data
            close_test: closing price test data
            prediction: predicted closing price for test range
            title: plot title
        Returns:
            displays plot
        """
        trace1 = go.Scatter(x = date_train, y = close_train, mode = 'lines', name = 'Training Data')
        trace2 = go.Scatter(x = date_val, y = close_val, mode = 'lines', name = 'Validation Data')
        trace3 = go.Scatter(x = date_test, y = prediction, mode = 'lines', name = 'Prediction')
        trace4 = go.Scatter(x = date_test, y = close_test, mode='lines', name = 'Test Ground Truth')
        layout = go.Layout(title = title, xaxis = {'title' : "Date"}, yaxis = {'title' : "Close Price ($USD"})
        fig = go.Figure(data=[trace1, trace2, trace3, trace4], layout=layout)
        fig.show()

    @staticmethod
    def plot_forecast(existing_dates, existing_prices, forecast_dates, forecast_prices, title):
        """
        Plots future forecast of stock price
        Params:
            existing_dates: real x data
            existing_prices: real y data
            forecast_dates: days into the future for the forecast
            forecast_prices: model-estimated prices for future period
            title: plot title
        Returns:
            displays plot
        """
        trace1 = go.Scatter(x = existing_dates, y = existing_prices, mode = 'lines', name = 'Data')
        trace2 = go.Scatter(x = forecast_dates, y = forecast_prices, mode = 'lines', name = 'Forecast')
        layout = go.Layout(title = title, xaxis = {'title' : "Date"}, yaxis = {'title' : "Close Price ($USD"})
        fig = go.Figure(data=[trace1, trace2], layout=layout)
        fig.show()
        
    @staticmethod
    def plot_model_comparison(model_names, model_scores, score_name):
        """
        Plots comparison of model scores
        Params:
            model_names: list of model names
            model_scores: list of model scores in same order as names
            score_name: name of the scoring metric
        Returns:
            displays bar graph
        """
        results = {'Model': model_names, score_name: model_scores}
        df = pd.DataFrame(results)
        fig, ax = plt.subplots(figsize=(18,12))
        ax.bar(height=df[score_name], x=df['Model'])
        ax.set_yticks(np.arange(10,110, step=10))
        fig.show()
