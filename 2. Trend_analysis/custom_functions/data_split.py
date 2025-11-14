import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter, AutoDateLocator

class DataSplitter:
    """A class for splitting and visualizing time series data into training and test sets."""

    def __init__(self, data, station_name):
        """
        Initialize with data and station name.
        """
        if not isinstance(data.index, pd.DatetimeIndex):
            data = data.copy()
            data.index = pd.to_datetime(data.index)
        self.data = data
        self.station = station_name
        self.train_data = None
        self.test_data = None

    # def to_timestamp(self,val):
    #         """Convert int/float → index position, str/datetime → Timestamp"""
    #         # Case 1: numeric → interpret as positional index
    #         if isinstance(val, (int, float)):
    #             return self.data.index[int(val)]

    #         # Case 2: string → try convert to pandas.Timestamp
    #         elif isinstance(val, str):
    #             try:
    #                 return pd.Timestamp(val)
    #             except ValueError:
    #                 raise ValueError(f"Cannot convert {val} to a Timestamp")

    #         # Case 3: datetime-like
    #         elif isinstance(val, pd.Timestamp):
    #             return val

    #         else:
    #             raise TypeError(f"Unsupported type for split boundary: {type(val)}")
    
    def compute_train_test_periods(self, min_test_days=None, manual_periods = None):
        """
        Compute train/test periods automatically based on the DataFrame index.
        Returns a dictionary with train_start, train_end, test_start, test_end (pd.Timestamp).
        """
        start_date = self.data.index.min()
        end_date = self.data.index.max()

        if min_test_days is None:
            min_test_days = 365

        # Test period at least min_test_days, max 1/3 of data
        test_days = min(min_test_days, max(1, (end_date - start_date).days // 3))
        test_start = end_date - pd.Timedelta(days=test_days - 1)
        train_start = start_date
        train_end = test_start - pd.Timedelta(days=1)

        # Combine into a single dictionary
        periods = {
            'train_start': train_start,
            'train_end': train_end,
            'test_start': test_start,
            'test_end': end_date
        }

        # --- Apply manual overrides ---
        if isinstance(manual_periods, dict):
            for key, value in manual_periods.items():
                if value is not None:
                    # Safely convert to Timestamp
                    try:
                        periods[key] = pd.Timestamp(value)
                    except Exception:
                        raise ValueError(f"Invalid date format for {key}: {value}")

        # --- logic to fill in any inconsistencies automatically ---
        # If train_end missing, infer from test_start
        if periods['train_end'] is None and periods['test_start'] is not None:
            periods['train_end'] = periods['test_start'] - pd.Timedelta(days=1)

        # If test_start missing, infer from train_end
        if periods['test_start'] is None and periods['train_end'] is not None:
            periods['test_start'] = periods['train_end'] + pd.Timedelta(days=1)

        # If train_start missing, use earliest date
        if periods['train_start'] is None:
            periods['train_start'] = start_date

        # If test_end missing, use latest date
        if periods['test_end'] is None:
            periods['test_end'] = end_date

        return periods

    def split(self, train_start, train_end, test_start, test_end):
        """
        Split data into training and testing sets based on date ranges.
        """

        # Split data
        self.train_data = self.data.loc[train_start:train_end]
        self.test_data = self.data.loc[test_start:test_end]


    def plot(self):
        """
        Plot training and test data side by side.
        """
        if self.train_data is None or self.test_data is None:
            raise ValueError("You must split the data before plotting.")

        fig, ax = plt.subplots(ncols=2, figsize=(10, 5), sharey=True, layout="constrained")

        # Plot training data
        ax[0].plot(self.train_data, color='blue')
        ax[0].set_title('Training Data')
        ax[0].set_ylabel('Temperature [°C]')
        loc1 = AutoDateLocator(minticks=5, maxticks=10)
        fmt1 = DateFormatter("%Y-%m")
        ax[0].xaxis.set_major_locator(loc1)
        ax[0].xaxis.set_major_formatter(fmt1)
        plt.setp(ax[0].xaxis.get_majorticklabels(), rotation=45, ha='right')

        # Plot test data
        ax[1].plot(self.test_data, color='orange')
        ax[1].set_title('Test Data')
        loc2 = AutoDateLocator(minticks=5, maxticks=10)
        fmt2 = DateFormatter("%Y-%m")
        ax[1].xaxis.set_major_locator(loc2)
        ax[1].xaxis.set_major_formatter(fmt2)
        plt.setp(ax[1].xaxis.get_majorticklabels(), rotation=45, ha='right')

        fig.suptitle(f"Data split of {self.station}", fontsize=14)
        fig.tight_layout()
        plt.show()

    def get_splits(self):
        """Return train and test data as DataFrames."""
        print(f"Training data dates \n start: {self.train_data.index[0].strftime('%Y-%m')} \n end: {self.train_data.index[-1].strftime('%Y-%m')} ")
        print(f"\n Test data dates \n start: {self.test_data.index[0].strftime('%Y-%m')} \n end: {self.test_data.index[-1].strftime('%Y-%m')} ")
        return self.train_data, self.test_data
