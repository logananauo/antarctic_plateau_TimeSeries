import numpy as np
import pandas as pd
import itertools
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.arima.model import ARIMA



split = int(len(series) * 0.80)
train = series.iloc[:split]
val = series.iloc[split:]
print(f"Training observations: {len(train)}")
print(f"Validation observations: {len(val)}")



def arima_grid_search_forecast(train, validation, p_range, d_range, q_range):

    results = []

    for p, d, q in itertools.product(p_range, d_range, q_range):
        try:
            ### fit ARIMA model
            model = ARIMA(train, order=(p, d, q)).fit()

            ### forecast validation period
            forecast = model.forecast(steps=len(validation))

            ### calculate metrics
            rmse = np.sqrt(mean_squared_error(validation, forecast))
            mae = mean_absolute_error(validation, forecast)

            ### store results
            results.append({
                'p': p,
                'd': d,
                'q': q,
                'RMSE': rmse,
                'MAE': mae,
                'AIC': model.aic,
                'BIC': model.bic
            })

        except Exception as e:
            print(f'ARIMA({p},{d},{q}) failed: {e}')

    results_df = pd.DataFrame(results)

    return results_df
