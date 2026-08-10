from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA




def arima_model(series, p, d, q):

    model = ARIMA(series, order=(p, d, q)).fit()
    residuals = model.resid
    
   # with sns.axes_style("darkgrid"):
        #with sns.color_palette("icefire"):
            
    fig, ax = plt.subplots(figsize=(6, 4))
    plot_acf(residuals, lags=20, ax=ax)
    ax.set_title('ACF of Residuals (Model Diagnostics)')
    plt.show()

    diag_plot = model.plot_diagnostics(figsize=(10, 8))
    plt.show()

    return {
        'model': print(f'Model: ARIMA({p, d, q})'),
        'summary': model.summary(),
        'acf_fig': fig,
        'diagnostics': diag_plot
    }

