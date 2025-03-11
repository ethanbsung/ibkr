import os
import pandas as pd
import numpy as np
from enum import Enum
from scipy.stats import norm

# ------------------- CONSTANTS & SETTINGS -------------------
DEFAULT_DATE_FORMAT = "%Y-%m-%d"
BUSINESS_DAYS_IN_YEAR = 256  # Trading days per year

# Parameters for screening
INITIAL_CAPITAL = 20000.0      # Your available capital
RISK_TARGET = 0.20              # e.g. 20% risk target
CONTRACTS_REQUIRED = 4          # Number of contracts used in the minimum-capital formula
# For risk estimation
USE_PERC_RETURNS = True         # Use percentage returns in volatility calculations
ANNUALISE_STDEV = True          # Annualise the volatility estimate

# ------------------- UPDATED pd_readcsv -------------------
def pd_readcsv(filename: str,
               date_format=DEFAULT_DATE_FORMAT,
               date_index_name: str = "Time") -> pd.DataFrame:
    """
    Reads a CSV file and parses the date column.
    Default date column is 'Time'. If not found, raises an error.
    """
    df = pd.read_csv(filename)
    if date_index_name not in df.columns:
        raise ValueError(f"Expected date column '{date_index_name}' not found in {filename}.")
    df.index = pd.to_datetime(df[date_index_name], format=date_format).values
    del df[date_index_name]
    df.index.name = None
    return df

# ------------------- HELPER FUNCTIONS FROM AUTHOR'S CODE -------------------
def calculate_daily_returns(adjusted_price: pd.Series) -> pd.Series:
    return adjusted_price.diff()

def calculate_percentage_returns(adjusted_price: pd.Series, current_price: pd.Series) -> pd.Series:
    daily_changes = calculate_daily_returns(adjusted_price)
    return daily_changes / current_price.shift(1)

def calculate_variable_standard_deviation_for_risk_targeting(adjusted_price: pd.Series,
                                                             current_price: pd.Series,
                                                             use_perc_returns: bool = True,
                                                             annualise_stdev: bool = True) -> pd.Series:
    if use_perc_returns:
        daily_returns = calculate_percentage_returns(adjusted_price, current_price)
    else:
        daily_returns = calculate_daily_returns(adjusted_price)
    
    daily_exp_std_dev = daily_returns.ewm(span=32).std()
    annualisation_factor = BUSINESS_DAYS_IN_YEAR ** 0.5 if annualise_stdev else 1
    annualised_std_dev = daily_exp_std_dev * annualisation_factor
    ten_year_vol = annualised_std_dev.rolling(BUSINESS_DAYS_IN_YEAR * 10, min_periods=1).mean()
    weighted_vol = 0.3 * ten_year_vol + 0.7 * annualised_std_dev
    
    return weighted_vol

class standardDeviation(pd.Series):
    def __init__(self,
                 adjusted_price: pd.Series,
                 current_price: pd.Series,
                 use_perc_returns: bool = True,
                 annualise_stdev: bool = True):
        stdev_series = calculate_variable_standard_deviation_for_risk_targeting(
            adjusted_price, current_price, use_perc_returns, annualise_stdev
        )
        super().__init__(stdev_series)
        self._use_perc_returns = use_perc_returns
        self._annualised = annualise_stdev
        self._current_price = current_price

    @property
    def annualised(self) -> bool:
        return self._annualised

    @property
    def use_perc_returns(self) -> bool:
        return self._use_perc_returns

    @property
    def current_price(self) -> pd.Series:
        return self._current_price

    def daily_risk_price_terms(self) -> pd.Series:
        stdev = self.copy()
        if self.annualised:
            stdev = stdev / (BUSINESS_DAYS_IN_YEAR ** 0.5)
        if self.use_perc_returns:
            stdev = stdev * self.current_price
        return stdev

    def annual_risk_price_terms(self) -> pd.Series:
        stdev = self.copy()
        if not self.annualised:
            stdev = stdev * (BUSINESS_DAYS_IN_YEAR ** 0.5)
        if self.use_perc_returns:
            stdev = stdev * self.current_price
        return stdev

def calculate_minimum_capital(multiplier: float,
                              price: float,
                              fx: float,
                              instrument_risk_ann_perc: float,
                              risk_target: float,
                              contracts: int = 4) -> float:
    """
    Calculates the minimum capital requirement using the formula:
      (contracts × multiplier × price × fx × σ%) ÷ risk_target
    """
    return contracts * multiplier * price * fx * instrument_risk_ann_perc / risk_target

# ------------------- INSTRUMENT SCREENING FUNCTION -------------------
def screen_instruments(symbols_csv: str,
                       initial_capital: float,
                       risk_target: float,
                       contracts: int = 4,
                       use_perc_returns: bool = True,
                       annualise_stdev: bool = True):
    """
    Reads a CSV file with columns 'Symbol' and 'Multiplier'
    and screens instruments based on the minimum capital requirement.
    """
    df_symbols = pd.read_csv(symbols_csv, comment='#')
    # Clean the Multiplier column (e.g. "0.1  # comment" becomes 0.1)
    df_symbols['Multiplier'] = df_symbols['Multiplier'].apply(lambda x: float(str(x).split()[0]))
    
    tradable = []
    non_tradable = []
    
    for _, row in df_symbols.iterrows():
        file_path = row['Symbol']
        multiplier = row['Multiplier']
        
        if not os.path.exists(file_path):
            print(f"File not found: {file_path} – skipping.")
            continue
        
        try:
            data = pd_readcsv(file_path)  # Uses the 'Time' column for dates
            data = data.dropna()
            
            # Use the author's expected columns if they exist;
            # otherwise, fall back to using "Last" as the price.
            if "adjusted" in data.columns and "underlying" in data.columns:
                adjusted_price = data["adjusted"]
                current_price = data["underlying"]
            elif "Last" in data.columns:
                adjusted_price = data["Last"]
                current_price = data["Last"]
            else:
                raise ValueError(f"Expected price columns not found in {file_path}")
            
            risk_series = standardDeviation(adjusted_price, current_price,
                                            use_perc_returns=use_perc_returns,
                                            annualise_stdev=annualise_stdev)
            risk_value = risk_series.iloc[-1]
            fx = 1.0
            latest_price = current_price.iloc[-1]
            
            min_cap_required = calculate_minimum_capital(multiplier=multiplier,
                                                         price=latest_price,
                                                         fx=fx,
                                                         instrument_risk_ann_perc=risk_value,
                                                         risk_target=risk_target,
                                                         contracts=contracts)
            
            instrument_info = {
                'File': file_path,
                'Multiplier': multiplier,
                'LatestPrice': latest_price,
                'Risk(σ%)': risk_value,
                'MinCapital': min_cap_required
            }
            
            if initial_capital >= min_cap_required:
                tradable.append(instrument_info)
            else:
                non_tradable.append(instrument_info)
                
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    return tradable, non_tradable

# ------------------- MAIN EXECUTION -------------------
if __name__ == "__main__":
    symbols_csv = "Data/symbols.csv"  # Update to your symbols CSV file path
    tradable, non_tradable = screen_instruments(symbols_csv,
                                                initial_capital=INITIAL_CAPITAL,
                                                risk_target=RISK_TARGET,
                                                contracts=CONTRACTS_REQUIRED,
                                                use_perc_returns=USE_PERC_RETURNS,
                                                annualise_stdev=ANNUALISE_STDEV)
    
    print("Tradable Instruments (min capital requirement <= available capital):")
    for instr in tradable:
        print(f"  {instr['File']} | Multiplier: {instr['Multiplier']} | Latest Price: {instr['LatestPrice']:.2f} | "
              f"Risk: {instr['Risk(σ%)']:.4f} | Min Capital: ${instr['MinCapital']:,.2f}")
    
    print("\nNon-Tradable Instruments (min capital requirement > available capital):")
    for instr in non_tradable:
        print(f"  {instr['File']} | Multiplier: {instr['Multiplier']} | Latest Price: {instr['LatestPrice']:.2f} | "
              f"Risk: {instr['Risk(σ%)']:.4f} | Min Capital: ${instr['MinCapital']:,.2f}")