import numpy as np
import pandas as pd
import itertools
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.arima.model import ARIMA


def arima_grid_search(train, validation, p_range, d_range, q_range):
    ''' Automates the fine-tuning parameter search for ARIMA time-series forecasting.
    Data must be split into training and validation sets. p, d, q ARIMA parameters
    may be entered as a range. The functions executes all possible combinations of
    p, d, q and produces a dataframe of each combination's RMSE, MAE, AIC, and BIC
    scores for the user to choose the best model based on performance metrics for 
    predictions and model complexity. '''

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



### ------------- RUNNING FUNCTION ------------- ###

### 80/20 split
split = int(len(series) * 0.80)
train = series.iloc[:split]
val = series.iloc[split:]
print(f"Training observations: {len(train)}")
print(f"Validation observations: {len(val)}")

### specifying range of ARIMA parameters; 4*3*4 = 48 total runs
results = arima_grid_search_forecast(
    train,
    val,
    p_range=range(0, 4),
    d_range=range(0, 3),
    q_range=range(0, 4)
)

### output performance results
results_sorted = results.sort_values(['RMSE', 'MAE']).reset_index(drop=True)
results_sorted.head(10)
