import pmdarima as pm
def train_and_predict(data=None):
    model = pm.auto_arima(data.values, seasonal=True)
    forecasts = model.predict(5)
    return forecasts
