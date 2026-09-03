import pandas as pd

data = pd.read_csv('apsrht_datetime.csv', index_col='date_time', parse_dates=True)
data = data.asfreq('15min')
hum_series = data['relative_hum_Ice']
temp_series = data['temp_C']
