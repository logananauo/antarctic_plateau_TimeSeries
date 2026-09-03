import pandas as pd
import itertools
from statsmodels.tsa.arima.model import ARIMA


def arima_grid_search_parameters(series, p_range, d_range, q_range):

    results = []

    for p, d, q in itertools.product(p_range, d_range, q_range):

        try:
            model = ARIMA(series, order=(p, d, q)).fit()

            results.append({
                'p': p,
                'd': d,
                'q': q,
                'AIC': model.aic,
                'BIC': model.bic
            })

        except Exception:
            print(f'ARIMA({p},{d},{q}) failed')

    results_df = pd.DataFrame(results)
    
    return results_df.sort_values('AIC').reset_index(drop=True)