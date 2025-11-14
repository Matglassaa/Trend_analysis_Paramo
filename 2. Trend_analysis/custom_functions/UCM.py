import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import statsmodels.api as sm
from matplotlib.dates import DateFormatter, AutoDateLocator
from sklearn.metrics import mean_absolute_error, mean_squared_error
from IPython.display import display

class UnobservedComponentsModel:
    """Encapsulates training, evaluation, and visualization of a UCM (Unobserved Components Model).
    
        Optional functions:
        .fit() -> fits a structural model based on the model components to the training data.
        ._extract_trend()
        ._plot_trend()
        .diagnostics()
        .forecast()
        """

    def __init__(self, train_data, model_settings, station, unit, test_data=None, n_steps=None):
        self.train_data = train_data
        self.test_data = test_data
        self.model_settings = model_settings
        self.station = station
        self.unit = unit
        self.n_steps = n_steps
        self.model = None
        self.results = None
        self.trend = None
        self.fitted = None
        self.forecast_mean = None
        self.forecast_ci = None

    def fit(self):
        """
        Fit the Unobserved Components Model to training data.
        
        Returns:    1) result of the model in the form of a summary of statistics
                    2) test statistic results as a dict -> Jarque-Bera, Ljung-Box and heteroskaticity
        """
        # Initiate model
        self.model = sm.tsa.UnobservedComponents(self.train_data, **self.model_settings)

        # Fit model to the data
        print(f"model: {self.model}")
        self.results = self.model.fit()
        print(self.results.summary())

        # Extract trend only if requested in model_settings
        if self.model_settings.get("trend", False):
            self._extract_trend()
        
        # Save test statistics to dictionary
        self.test_statistics = {"JB_pval": self.results.test_normality(method='jarquebera')[0][1],
                            "Q_pval": self.results.test_serial_correlation(method='ljungbox')[0][1][1],
                            "H_pval": self.results.test_heteroskedasticity(method='breakvar')[0][1]}
        
        return self.results, self.test_statistics


    def _extract_trend(self):
        """
        Compute and store the smoothed level (trend).
        """
        # Get trend
        self.trend = pd.Series(self.results.level.smoothed, index=self.train_data.index)
        trend_slope = (self.trend.iloc[-1] - self.trend.iloc[0]) / len(self.trend)          # Compute slope
        trend_per_year = trend_slope * 365                                                  # Compute slope per year

        # If trend is requested, only print the result if it is significant
        #if trend_per_year > 0.01:               #This should be different, 0.01 only relates to temperature
        print(f"Average trend (°C per day): {trend_slope:.4f}")
        print(f"Average trend (°C per year): {trend_per_year:.3f}")

        #Plot trend if conditions are met
        self._plot_trend()


    def _plot_trend(self):
        """
        Plot the smoothed trend alongside observed data.
        """
        fig, ax = plt.subplots(figsize=(15, 5))
        ax.plot(self.trend, label='Modeled Level')
        ax.plot(self.train_data, label='Observations', color='grey')
        ax.legend()
        ax.yaxis.grid()
        plt.title(f"Temperature observations and modeled level - {self.station}")
        ax.set_xlabel('Date')
        ax.set_ylabel(f"Temperature [{self.unit}]")
        plt.show()


    def diagnostics(self):
        """
        Run diagnostic tests and plot residual analysis.
        """
        self.results.plot_diagnostics(figsize=(13, 8))
        plt.show()


    def forecast(self, exog_future=None, bins=None):
        """
        Function that accomplished the forecasting part if requested.

        Returns:    1) Array of fitted values to the training data
                    2) The forecast mean of the model
                    3) dictionary of a summary of the performace w.r.t. the test data
        """
        if self.test_data is None and self.n_steps is None:
            raise ValueError("Either test_data or n_steps must be provided for forecasting.")
        if exog_future is not None:
            steps = len(exog_future)
        else: 
            steps = len(self.test_data)
        print(f"results: {self.results}")

        self.fitted = self.results.fittedvalues

        forecast_res = self.results.get_forecast(steps=steps, exog=exog_future)
        self.forecast_mean = forecast_res.predicted_mean
        self.forecast_ci = forecast_res.conf_int()

        if self.test_data is not None and len(self.test_data) > 0:
            # If test data is provided → use its index to align forecast results
            forecast_index = self.test_data.index
        else:
            print('error')
        # Assign forecast index
        self.forecast_mean.index = forecast_index
        self.forecast_ci.index = forecast_index

        # Evaluate the forecast by testing the fitted values vs the test data -> dropnan vlaues, allign axis and subtract
        test_data_not_Nan = self.test_data.dropna(axis=0)
        aligned_forecast, aligned_test = self.forecast_mean.align(test_data_not_Nan, join="inner", axis=0)
        if isinstance(aligned_test, pd.DataFrame):
            aligned_test_series = aligned_test.iloc[:, 0]
        else:
            aligned_test_series = aligned_test
        self.test_residuals = aligned_test_series - aligned_forecast

        #Train residuals
        aligned_fitted, aligned_train = self.fitted.align(self.train_data, join="inner", axis=0)
        if isinstance(aligned_train, pd.DataFrame):
            aligned_train_series = aligned_train.iloc[:, 0]
        else:
            aligned_train_series = aligned_train
        self.train_residuals = aligned_train_series - aligned_fitted

        # Compute statistics of resduals
        mae = mean_absolute_error(aligned_test, aligned_forecast)
        rmse = np.sqrt(mean_squared_error(aligned_test, aligned_forecast))
        mape = (np.abs(self.test_residuals) / aligned_test_series).mean() * 100  # if data > 0
        var_res = self.test_residuals.var()

        # Plot histogram of residuals
        n_bins = bins if bins is not None else 30

        fig,ax = plt.subplots(figsize=(10,5))
        ax = sb.histplot(data= self.test_residuals, bins = n_bins, kde= True)
        ax.set_title(f"Residuals test values vs forecasted values {self.station}")
        ax.set_xlabel(f"{self.unit}")

        # Add text box with mean and variance
        text_str = (f"MAE: {mae:.3f}\n"
                    f"RMSE: {rmse:.3f}\n"
                    f"MAPE: {mape:.2f}%"
                    f"Var: {var_res:.3f}\n"
                    )
        ax.text(
            0.98, 0.95, text_str,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='top',
            horizontalalignment='right',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7)
        )
        plt.tight_layout()
        display(fig)
        plt.close()

        # Group residual values into dictionary
        residuals_overview = {'test_residuals':self.test_residuals,
                            'train_residuals':self.train_residuals}

        return self.fitted, self.forecast_mean, residuals_overview

    def plot_results(self, plot_trend=None):
        """
        Plot fitted, forecast, and observed data.
        """
        if self.forecast_mean is None or self.forecast_ci is None:
            self.forecast()  # ensure forecast exists

        n_obs_init = self.model.k_states - int(self.model._unused_state) - self.model.ar_order

        fig, ax = plt.subplots(figsize=(12, 6))
        fig.suptitle(f"UCM Predictions - {self.station}", fontsize=12)

        # Plot training and test data
        ax.plot(self.train_data, label='Training Data',alpha=0.5)
        if self.test_data is not None:
            ax.plot(self.test_data, label='Test Data',alpha=0.5)

        # Plot fitted and trend
        ax.plot(self.fitted, label='Fitted', linestyle='--', color='green')
        if plot_trend == True:
            ax.plot(self.trend, label='Trend', color='black')
        # if self.trend is not None:
        #     ax.plot(self.trend, label='Trend', color='black')

        # 🔧 Plot forecast and CI
        if self.forecast_mean is not None:
            ax.plot(self.forecast_mean.index, self.forecast_mean, label='Forecast', color='red')
            ax.fill_between(
                self.forecast_ci.index,
                self.forecast_ci.iloc[:, 0],
                self.forecast_ci.iloc[:, 1],
                color='pink', alpha=0.5, label='95% CI'
            )

        # Split markers
        #ax.axvline(x=self.train_data.index[n_obs_init], color="C5", linestyle="--", label="Diffuse Init")
        #ax.axvline(x=self.train_data.tail(1).index[0], color="C6", linestyle="--", label="Train-Test Split")
        # Split markers (no legend entries)
        ax.axvline(x=self.train_data.index[n_obs_init], color="black", linestyle="-.", linewidth=1)
        ax.text(self.train_data.index[n_obs_init], ax.get_ylim()[1]*0.95, "Diffuse Init",
                rotation=90, color="black", va="top", ha="right", fontsize=11)

        ax.axvline(x=self.test_data.head(1).index[0], color="black", linestyle="-.", linewidth=1)
        ax.text(self.test_data.head(1).index[0], ax.get_ylim()[1]*0.95, "Train-Test Split",
                rotation=90, color="black", va="top", ha="right", fontsize=11)



        ax.grid(which='major', linestyle='-', linewidth='0.5', color='gray')
        ax.grid(which='minor', linestyle=':', linewidth='0.25', color='lightgray')
        ax.legend()
        ax.set_ylabel(f"{self.unit}")
        ax.set_xlabel('Date')
        plt.show()

    def report_scores(self):
        """
        Display key model selection statistics as a formatted table.
        """
        scores = {
            "AIC": [self.results.aic],
            "BIC": [self.results.bic],
            "HQIC": [self.results.hqic],
            "Log-Likelihood": [self.results.llf],
            "Total Score": [self.results.aic + self.results.bic + self.results.hqic + self.results.llf],
        }

        df_scores = pd.DataFrame(scores).T  # transpose for vertical layout
        df_scores.columns = ["Value"]
        df_scores["Value"] = df_scores["Value"].round(3)
        print("\nModel Evaluation Metrics:\n")
        print(df_scores)  # Jupyter/IPython display
        return df_scores