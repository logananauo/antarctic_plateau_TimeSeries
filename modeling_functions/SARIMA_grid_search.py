import itertools
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error


def sarima_grid_search(train, validation, p_range, d_range, q_range, P_range, D_range, Q_range, seasonal_period=96):
    ''' Automates the fine-tuning parameter search for SARIMA time-series forecasting.
    Data must be split into training and validation sets. Parameters
    may be entered as a range. The functions executes all possible combinations of
    [p, d, q, P, D, Q, seasonal period] and produces a dataframe of each combination's 
    RMSE, MAE, AIC, and BIC scores for the user to choose the best model based on performance 
    metrics for predictions and model complexity. '''
    
    results = []

    ### generate all parameter combinations
    combinations = itertools.product(
        p_range,
        d_range,
        q_range,
        P_range,
        D_range,
        Q_range
    )

    for p, d, q, P, D, Q in combinations:
        try:
            ### fit SARIMA model
            model = SARIMAX(
                train,
                order=(p, d, q),
                seasonal_order=(P, D, Q, seasonal_period),
                enforce_stationarity=False,
                enforce_invertibility=False
            ).fit(disp=False)

            ### forecast validation period
            forecast = model.forecast(steps=len(validation))

            ### calculate metrics
            rmse = np.sqrt(mean_squared_error(validation, forecast))
            mae = mean_absolute_error(validation, forecast)

            results.append({
                'p': p,
                'd': d,
                'q': q,
                'P': P,
                'D': D,
                'Q': Q,
                'seasonal_period': seasonal_period,
                'RMSE': rmse,
                'MAE': mae,
                'AIC': model.aic,
                'BIC': model.bic
            })

        except Exception as e:
            print(f'SARIMA({p},{d},{q})'f'({P},{D},{Q},{seasonal_period}) failed')

    results_df = pd.DataFrame(results)
    return results_df



### -------------- RUNNING FUNCTION -------------- ###

### 80/20 split
split = int(len(series) * 0.80)
temp_train = series.iloc[:split]
temp_val = series.iloc[split:]

### call function; input parameters
results = sarima_grid_search(
    train,
    val,
    p_range=range(0, 3),
    d_range=range(0, 2),
    q_range=range(0, 3),
    P_range=range(0, 2),
    D_range=range(0, 2),
    Q_range=range(0, 2),
    seasonal_period=96
)

### output performance metrics
results_sorted = results.sort_values('RMSE').reset_index(drop=True)
results_sorted.head()
