import gc
import traceback
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from loguru import logger
from matplotlib.ticker import FixedLocator, FixedFormatter
from scipy.stats import norm
from statsmodels.tsa.stattools import acf

def diebold_mariano_test(errors, naive_errors, h=1, alpha=0.05):
    # calc differences
    d = errors - naive_errors
    mean_d = np.mean(d)
    var_d = np.var(d, ddof=1)
    n = len(d)

    # serial correlation
    acf_values = acf(d, nlags=h - 1, fft=False)
    dm_stat = mean_d / np.sqrt((var_d / n) * (1 + 2 * np.sum(acf_values)))

    # p-values
    p_value = 2 * (1 - norm.cdf(np.abs(dm_stat)))
    significant = '*' if p_value < alpha else ''
    return dm_stat, p_value, significant

def giacomini_white_test(errors, naive_errors, X, alpha=0.05):
    # calc differences
    d = errors - naive_errors
    n = len(d)

    X = np.column_stack([np.ones(n), X])
    beta_hat = np.linalg.inv(X.T @ X) @ X.T @ d

    # residuals/variances
    residuals = d - X @ beta_hat
    sigma_hat = np.var(residuals, ddof=X.shape[1])

    # cov matrix
    cov_matrix = sigma_hat * np.linalg.inv(X.T @ X)
    test_stat = beta_hat[-1] / np.sqrt(cov_matrix[-1, -1])

    # p-values
    p_value = 2 * (1 - norm.cdf(np.abs(test_stat)))
    significant = '*' if p_value < alpha else ''
    return test_stat, p_value, significant

def reconstruct_levels(target_vector, results):
    reconstructed_forecasts = []
    reconstructed_actuals = []

    # initial level from the target vector
    initial_level = target_vector.iloc[0]

    # levels for each forecast and actual by reversing the log differencing
    for i, date in enumerate(results['dates']):
        current_index = target_vector.index.get_loc(date)
        previous_level = initial_level if i == 0 else reconstructed_actuals[-1]

        # exponentiate
        reconstructed_forecast = np.exp(results['forecasts'][i]) * previous_level
        reconstructed_actual = np.exp(results['actuals'][i]) * previous_level
        reconstructed_forecasts.append(reconstructed_forecast)
        reconstructed_actuals.append(reconstructed_actual)

    return {
        'forecasts': reconstructed_forecasts,
        'actuals': reconstructed_actuals,
        'dates': results['dates']
    }

def rolling_test(model_class, feature_matrix, feature_vector, chunk, predictions, sequence_length=None, model_name=""):
    logger.info(f'Starting {model_name} test')
    forecasts, actuals, forecast_dates = [], [], []
    t = 0

    feature_matrix = feature_matrix.astype(np.float32)
    feature_vector = feature_vector.astype(np.float32)

    while t < predictions:
        end_idx = -(predictions - t) if t != predictions - 1 else None
        start_idx = -(chunk - t) if t < chunk else None

        X_fit = feature_matrix.iloc[start_idx:end_idx]
        y_fit = feature_vector.iloc[start_idx:end_idx]
        X_test = feature_matrix.iloc[end_idx:end_idx+1] if end_idx is not None else feature_matrix.iloc[-1:]
        y_test = feature_vector.iloc[end_idx:end_idx+1] if end_idx is not None else feature_vector.iloc[-1:]

        if y_test.empty:
            logger.info(f"No data for {t}")
            t += 1
            del X_fit, y_fit, X_test, y_test
            gc.collect()
            continue

        try:
            model = model_class(sequence_length=sequence_length)
            model.fit(X_fit.values, y_fit.values)
            logger.info(f'Test {t+1}/{predictions}')
            forecast = model.predict().flatten()

            forecasts.append(round(forecast[0], 4))
            actuals.append(round(y_test.values.flatten()[0], 4))
            forecast_dates.append(y_test.index[0])

        except Exception as e:
            logger.error(f"An error occurred: {traceback.format_exc()}")

        finally:
            if hasattr(model, 'cleanup'):
                model.cleanup()
            del model, X_fit, y_fit, X_test, y_test
            tf.keras.backend.clear_session()
            gc.collect()

        t += 1

    return {
        'forecasts': forecasts,
        'actuals': actuals,
        'dates': forecast_dates
    }

class ForecastStats:
    def __init__(self, raw_metrics, X):
        self.raw_metrics = raw_metrics
        self.X = X

    def calculate_forecast_stats(self):
        naive_errors_raw = np.array(self.raw_metrics['Naïve Forecast']['actuals']) - np.array(self.raw_metrics['Naïve Forecast']['forecasts'])
        table_raw = pd.DataFrame(columns=['Model', 'MAE', 'RMSE'])
        rows = []

        # naive calc'd seperately
        naive_mae = np.mean(np.abs(naive_errors_raw))
        naive_rmse = np.sqrt(np.mean(naive_errors_raw**2))

        rows.append({
            'Model': 'Naïve Forecast',
            'MAE': naive_mae,
            'RMSE': naive_rmse
        })

        # calc metrics for models
        for model_name, metrics in self.raw_metrics.items():
            if model_name == 'Naïve Forecast':
                continue

            errors = np.array(metrics['actuals']) - np.array(metrics['forecasts'])
            mae = np.mean(np.abs(errors))
            rmse = np.sqrt(np.mean(errors**2))

            # significance tests
            dm_stat, dm_p_value, dm_sig = diebold_mariano_test(errors, naive_errors_raw)
            gw_stat, gw_p_value, gw_sig = giacomini_white_test(errors, naive_errors_raw, self.X.iloc[-len(errors):])

            # populate
            rows.append({
                'Model': model_name,
                'MAE': mae,
                'RMSE': rmse,
                'Diebold-Mariano': f"{dm_stat:.2f} ({dm_p_value:.3f}){dm_sig}",
                'Giacomini-White': f"{gw_stat:.2f} ({gw_p_value:.3f}){gw_sig}"
            })

        table_raw = pd.DataFrame(rows)
        return table_raw

    def calculate_level_metrics(self, levels_metrics):
        table_levels = pd.DataFrame(columns=['Model', 'MAPE', 'Log Loss'])
        rows = []

        # calc stats for levels
        naive_actuals = np.array(levels_metrics['Naïve Forecast']['actuals'])
        naive_forecasts = np.array(levels_metrics['Naïve Forecast']['forecasts'])
        naive_mape = np.mean(np.abs((naive_actuals - naive_forecasts) / naive_actuals)) * 100
        naive_log_loss = np.mean((np.log(naive_actuals / naive_forecasts))**2)

        rows.append({
            'Model': 'Naïve Forecast',
            'MAPE': naive_mape,
            'Log Loss': naive_log_loss,
        })

        # calc metrics for models
        for model_name, metrics in levels_metrics.items():
            if model_name == 'Naïve Forecast':
                continue

            actuals = np.array(metrics['actuals'])
            forecasts = np.array(metrics['forecasts'])
            mape_value = np.mean(np.abs((actuals - forecasts) / actuals)) * 100
            log_loss_value = np.mean((np.log(actuals / forecasts))**2)

            # populate
            rows.append({
                'Model': model_name,
                'MAPE': mape_value,
                'Log Loss': log_loss_value,
            })

        table_levels = pd.concat([table_levels, pd.DataFrame(rows)], ignore_index=True)
        return table_levels

def plot_predictions_vs_actuals(metrics, models, model_name_suffix, window):
    fig_predictions, axes_predictions = plt.subplots(1, 3, figsize=(15, 5))

    for i, model in enumerate(models):
        suffix = f'{model}{model_name_suffix}'
        dates = metrics[suffix]['dates']
        forecasts = metrics[suffix]['forecasts']
        actuals = metrics[suffix]['actuals']

        axes_predictions[i].plot(dates, actuals, 'o', color='grey', label='Actual')
        axes_predictions[i].plot(dates, forecasts, 'x', color='black', label='Forecast')
        axes_predictions[i].set_title(f'{model} - Predictions vs Actuals (Levels)')
        axes_predictions[i].set_xlabel('Date')
        axes_predictions[i].set_ylabel('Value')
        axes_predictions[i].legend()

        axes_predictions[i].xaxis.set_major_formatter(mdates.DateFormatter('%m/%y'))
        axes_predictions[i].xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.setp(axes_predictions[i].xaxis.get_majorticklabels(), rotation=45)

    plt.tight_layout()
    fig_predictions.savefig(f'predictions_vs_actuals_plots_{window}-period.png')
    plt.show()

def plot_residuals(metrics, models, model_name_suffix, window):
    fig_residuals, axes_residuals = plt.subplots(1, 3, figsize=(15, 5))
    all_residuals = []
    all_residuals_flat = []

    for i, model in enumerate(models):
        suffix = f'{model}{model_name_suffix}'
        dates = metrics[suffix]['dates']
        residuals = np.array(metrics[suffix]['actuals']) - np.array(metrics[suffix]['forecasts'])
        all_residuals.append(residuals)
        all_residuals_flat.extend(residuals)

        axes_residuals[i].plot(dates, residuals, 'x', color='black')
        axes_residuals[i].axhline(y=0, color='grey', linestyle='--')
        for date, residual in zip(dates, residuals):
            axes_residuals[i].vlines(x=date, ymin=0, ymax=residual, color='black', linestyle='--', alpha=0.5)
        axes_residuals[i].set_title(f'{model} - Residuals')
        axes_residuals[i].set_xlabel('Date')
        axes_residuals[i].set_ylabel('Residual')

        axes_residuals[i].xaxis.set_major_formatter(mdates.DateFormatter('%m/%y'))
        axes_residuals[i].xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.setp(axes_residuals[i].xaxis.get_majorticklabels(), rotation=45)

    global_min = min(all_residuals_flat)
    global_max = max(all_residuals_flat)
    padding = (global_max - global_min) * 0.05
    global_min_padded = global_min - padding
    global_max_padded = global_max + padding

    for i in range(3):
        axes_residuals[i].set_ylim(global_min_padded, global_max_padded)

    plt.tight_layout()
    fig_residuals.savefig(f'residuals_plots_{window}-period.png')
    plt.show()

def plot_residuals_violin(all_residuals, models, window):
    fig_violin, ax_violin = plt.subplots(figsize=(15, 4))

    sns.violinplot(
        data=all_residuals,
        orient='h',
        ax=ax_violin,
        palette=['#f0f0f0']*len(models),
        inner=None,
        linewidth=0
    )

    sns.boxplot(
        data=all_residuals,
        orient='h',
        ax=ax_violin,
        showcaps=True,
        width=0.2,
        boxprops={'facecolor':'none', 'edgecolor':'black', 'linewidth':0.5},
        whiskerprops={'color':'black', 'linewidth':0.5},
        capprops={'color':'black', 'linewidth':0.5},
        medianprops={'color':'black', 'linewidth':0.5},
        flierprops={'marker':'o', 'color':'black', 'markersize':1, 'linestyle':'none'}
    )

    ax_violin.yaxis.set_major_locator(FixedLocator([0, 1, 2]))
    ax_violin.yaxis.set_major_formatter(FixedFormatter(models))

    ax_violin.set_title(f'Variance of Errors, {window}-period window')
    ax_violin.set_xlabel('Error')

    plt.tight_layout()
    fig_violin.savefig('residuals_violin_plots.png')
    plt.show()
