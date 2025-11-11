import os
import json
import logging
import warnings
import itertools
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kwargs):
        return x

warnings.filterwarnings("ignore")


# --- Cointegration config (simple, robust defaults) ---
# Formation window used to IDENTIFY pairs at selection time (1–2 years)
COINT_FORMATION_WINDOW_DAYS = 504  # 2y; set to 252 for 1y

# Re-test during trading
COINT_RETEST_WINDOW_DAYS = 504     # 2y rolling residuals, aligns with identification
COINT_RETEST_FREQ = 'monthly'      # first business day each month
COINT_SUSPEND_P = 0.10             # suspend trading if p >= 0.10
COINT_REACTIVATE_P = 0.05          # reactivate if p < 0.05
COINT_MIN_POINTS = 126             # minimum data points to run a valid test


CD_cluster_df = pd.read_pickle(f"./data/Final Results/year_cluster_dfC-CD.pkl")
CD_cluster_df["year"] = CD_cluster_df["year"].astype(int)
CD_cluster_df = CD_cluster_df[~CD_cluster_df['year'].isin([1993, 1994, 1995])].reset_index(drop=True)

Rolling_CD_cluster_df = pd.read_pickle(f"./data/Final Results/year_cluster_dfrollingCD.pkl")
Rolling_CD_cluster_df["year"] = Rolling_CD_cluster_df["year"].astype(int)
Rolling_CD_cluster_df = Rolling_CD_cluster_df[~Rolling_CD_cluster_df['year'].isin([1993, 1994, 1995])].reset_index(drop=True)

TPG_Cluster_df = pd.read_pickle(f"./data/Final Results/year_cluster_dfPaLM-gecko.pkl")
TPG_Cluster_df["year"] = TPG_Cluster_df["year"].astype(int)
TPG_Cluster_df = TPG_Cluster_df[~TPG_Cluster_df['year'].isin([1993, 1994, 1995])].reset_index(drop=True)

SBERT_Cluster_df = pd.read_pickle(f"./data/Final Results/year_cluster_dfSBERT.pkl")
SBERT_Cluster_df["year"] = SBERT_Cluster_df["year"].astype(int)
SBERT_Cluster_df = SBERT_Cluster_df[~SBERT_Cluster_df['year'].isin([1993, 1994, 1995])].reset_index(drop=True)

BERT_Cluster_df = pd.read_pickle(f"./data/Final Results/year_cluster_dfBERT.pkl")
BERT_Cluster_df["year"] = BERT_Cluster_df["year"].astype(int)
BERT_Cluster_df = BERT_Cluster_df[~BERT_Cluster_df['year'].isin([1993, 1994, 1995])].reset_index(drop=True)

year_SIC_cluster_df = pd.read_pickle("./data/cointegration/year_SIC_cluster_mapping.pkl")
year_Industry_cluster_df = pd.read_pickle("./data/cointegration/year_Industry_cluster_mapping.pkl")
time_series_data = pd.read_pickle("./data/cointegration/cik_ticker_timeseries.pkl")
with open("./data/cointegration/index_cik_ticker_map.json", "r") as json_file:
    index_cik_ticker_map = json.load(json_file)


# Function to get price series
def get_price_series(company_index, time_series_data, start_date=None, end_date=None):
    """
    Fetch the price series for a given company index, filtered by start and end dates.
    """
    if str(company_index) not in index_cik_ticker_map:
        return None
    tickers = index_cik_ticker_map[str(company_index)]["ticker"]

    for ticker in tickers:
        ticker_data = time_series_data[time_series_data['ticker'] == ticker]
        if not ticker_data.empty:
            timeseries = ticker_data.iloc[0]['timeseries']
            timeseries = pd.Series(timeseries)
            timeseries.index = pd.to_datetime(timeseries.index)

            # Filter by date range if specified
            if start_date and end_date:
                timeseries = timeseries[start_date:end_date]
            return timeseries.sort_index()
    return None

def identify_and_save_cointegrated_pairs(
    cluster_type,
    cluster_df,
    time_series_data,
    year,
    correlation_threshold=0.95,
    top_n=10000000,
    formation_window_days: int = 504,
):
    """
    Identify and rank cointegrated pairs for each cluster type by p-value, saving the top N pairs.
    - Membership: ONLY the selection year's clusters (e.g., 2013)
    - Test window: Last `formation_window_days` ending at {year}-12-31 (1–2 years)
    - Filter: Prefilter by correlation on the same window, then Engle–Granger (ADF on residuals)
    """
    # Using module-level imports for itertools, adfuller, and tqdm

    # Determine the years to include: use trailing 3-year membership (e.g., 2011, 2012, 2013)
    available_years = set(cluster_df['year'].astype(int).unique())
    candidate_years = [year - 2, year - 1, year]
    years_to_include = [y for y in candidate_years if y in available_years]

    # Select clusters for the specified years
    clusters_list = cluster_df.loc[cluster_df['year'].isin(years_to_include), 'clusters'].values

    # Combine clusters from different years
    combined_clusters = {}
    for clusters in clusters_list:
        for cluster_id, companies in clusters.items():
            # Create a unique key for each cluster to avoid ID conflicts
            key = f"{cluster_id}_{cluster_df.loc[cluster_df['clusters'] == clusters, 'year'].values[0]}"
            if key in combined_clusters:
                combined_clusters[key].extend(companies)
                # Remove duplicates
                combined_clusters[key] = list(set(combined_clusters[key]))
            else:
                combined_clusters[key] = companies.copy()

    cointegrated_pairs = []

    # Initialize tqdm for clusters with total number of clusters
    total_clusters = len(combined_clusters)
    cluster_iterator = tqdm(combined_clusters.items(), desc=f"Processing Clusters for {cluster_type}", total=total_clusters)

    for cluster_id, companies in cluster_iterator:
        if len(companies) < 2:
            continue

        # Generate all possible pairs within the cluster
        pair_combinations = list(itertools.combinations(companies, 2))
        total_pairs = len(pair_combinations)

        # Initialize tqdm for pairs within the current cluster
        pair_iterator = tqdm(pair_combinations, desc=f"Cluster {cluster_id}", leave=False, total=total_pairs)

        for company1, company2 in pair_iterator:
            # Fetch price series up to selection year end, then restrict to last `formation_window_days`
            end_date = f"{year}-12-31"
            # pull wider then window with an early start but we'll trim by tail
            # Pull a broad span then restrict to trailing 2-year window
            series1 = get_price_series(company1, time_series_data, start_date="2010-01-01", end_date=end_date)
            series2 = get_price_series(company2, time_series_data, start_date="2010-01-01", end_date=end_date)
            if series1 is None or series2 is None:
                continue

            # Align lengths and dates
            combined_df = pd.DataFrame({'series1': series1, 'series2': series2}).dropna()
            # Use only the trailing formation window
            if formation_window_days is not None and formation_window_days > 0 and len(combined_df) > formation_window_days:
                combined_df = combined_df.tail(formation_window_days)
            if len(combined_df) < COINT_MIN_POINTS:
                continue

            # Check if all values in series are identical
            if combined_df['series1'].nunique() <= 1 or combined_df['series2'].nunique() <= 1:
                continue

            # Calculate correlation
            correlation = combined_df['series1'].corr(combined_df['series2'])

            # Check correlation threshold
            if abs(correlation) < correlation_threshold:
                continue  # Skip pairs with low correlation

            # Engle–Granger step 1: OLS hedge ratio on training window
            y = combined_df['series1']
            X = sm.add_constant(combined_df['series2'])
            try:
                eg_model = sm.OLS(y, X, missing='drop').fit()
                resid = y - eg_model.predict(X)
            except Exception:
                continue

            # Engle–Granger step 2: ADF on residuals
            try:
                adf_result = adfuller(resid.dropna())
            except Exception:
                continue
            p_value = adf_result[1]

            if p_value < 0.05:  # p-value < 0.05 indicates stationarity (cointegration)
                cointegrated_pairs.append({
                    'Company1': company1,
                    'Company2': company2,
                    'ClusterID': cluster_id,
                    'Correlation': correlation,
                    'ADFStat': adf_result[0],
                    'p-value': p_value
                })

    print("Identified # of cointegrated pairs:", len(cointegrated_pairs))
    # Convert to DataFrame
    cointegrated_pairs_df = pd.DataFrame(cointegrated_pairs)

    # Sort by p-value and select top N pairs
    try:
        cointegrated_pairs_df = cointegrated_pairs_df.sort_values(by='p-value').head(top_n)
    except:
        print(cointegrated_pairs_df)
    # Save to CSV (consistent path with loader)
    out_path = f'./data/cointegration/cointegrated_pairs_{cluster_type}.csv'
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cointegrated_pairs_df.to_csv(out_path, index=False)
    print(f"Saved top {top_n} cointegrated pairs for {cluster_type} to {out_path}")

    return cointegrated_pairs_df

# (Removed redundant duplicate imports below; using module-level imports declared above)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Diagnostics holders (for external introspection/plots) ---
traders = {}
traded_pairs_by_cluster = {}
residual_std_summary = {}

class Position:
    def __init__(self, pair, entry_date, position_type, units1, units2, entry_price1, entry_price2):
        """
        Initialize a Position.

        Parameters:
        - pair: Tuple identifying the pair (e.g., (Company1, Company2))
        - entry_date: Date when the position was opened
        - position_type: 'long_spread' or 'short_spread'
        - units1: Units of Asset 1 (positive for buy, negative for sell)
        - units2: Units of Asset 2 (positive for buy, negative for sell)
        - entry_price1: Entry price of Asset 1
        - entry_price2: Entry price of Asset 2
        """
        self.pair = pair
        self.entry_date = entry_date
        self.position_type = position_type  # 'long_spread' or 'short_spread'
        self.units1 = units1
        self.units2 = units2
        self.entry_price1 = entry_price1
        self.entry_price2 = entry_price2
        self.exit_date = None
        self.exit_price1 = None
        self.exit_price2 = None
        self.pnl = 0
        # Track the initial capital required for this position
        self.initial_value = abs(units1 * entry_price1) + abs(units2 * entry_price2)

    def get_mtm_value(self, current_price1, current_price2):
        """
        Calculate the mark-to-market value of the position.
        
        Returns the unrealized P&L based on current prices.
        """
        unrealized_pnl1 = self.units1 * (current_price1 - self.entry_price1)
        unrealized_pnl2 = self.units2 * (current_price2 - self.entry_price2)
        return unrealized_pnl1 + unrealized_pnl2

    def get_current_market_value(self, current_price1, current_price2):
        """
        Get the current market value of the position (not P&L, but actual value).
        """
        return self.units1 * current_price1 + self.units2 * current_price2

    def close(self, exit_date, exit_price1, exit_price2):
        """
        Close the position and calculate PnL.

        Parameters:
        - exit_date: Date when the position was closed
        - exit_price1: Exit price of Asset 1
        - exit_price2: Exit price of Asset 2

        Returns:
        - pnl: Profit and Loss from the position
        """
        self.exit_date = exit_date
        self.exit_price1 = exit_price1
        self.exit_price2 = exit_price2

        # Calculate PnL for Asset 1
        pnl1 = self.units1 * (self.exit_price1 - self.entry_price1)

        # Calculate PnL for Asset 2
        pnl2 = self.units2 * (self.exit_price2 - self.entry_price2)

        self.pnl = pnl1 + pnl2
        return self.pnl

    def __repr__(self):
        return (f"Position(pair={self.pair}, entry_date={self.entry_date.date()}, "
                f"type={self.position_type}, units1={self.units1:.4f}, units2={self.units2:.4f})")

class PairTrader:
    def __init__(self, capital_per_pair=10000, delta=1.0, initial_capital=100000, index_cik_ticker_map=None,
                 coint_retest_window_days: int = COINT_RETEST_WINDOW_DAYS,
                 coint_suspend_p: float = COINT_SUSPEND_P,
                 coint_reactivate_p: float = COINT_REACTIVATE_P,
                 coint_retest_freq: str = COINT_RETEST_FREQ,
                 # Guard-rails
                 min_price: float = 1.0,
                 max_price: float = 5000.0,
                 resid_std_floor: float = 1e-6,
                 z_score_cap: float = 8.0,
                 max_abs_leg_return: float = 0.50,
                 # WATCH sizing
                 watch_scale: float = 0.5,
                 # Data coverage requirement toggle
                 require_full_tail: bool = False,
                 tail_date: str = "2020-12-01"):
        """
        Initialize the PairTrader with proper MTM tracking.

        Parameters:
        - capital_per_pair: Capital allocated per pair trade
        - delta: Entry threshold (δ) for z-score
        - initial_capital: Starting cash for the portfolio
        - index_cik_ticker_map: Mapping from company indices to tickers
        """
        self.capital_per_pair = capital_per_pair
        self.delta = delta  # Entry threshold (δ)
        self.positions = []  # All closed positions
        self.open_positions = []  # Currently open positions
        self.cumulative_pnl = 0
        self.total_trades = 0
        
        # Daily tracking series
        self.daily_pnl = defaultdict(float)  # Daily realized P&L
        self.daily_unrealized_pnl = defaultdict(float)  # Daily unrealized P&L
        self.daily_portfolio_value = defaultdict(float)  # Daily total portfolio value
        self.daily_cash = defaultdict(float)  # Daily cash position
        
        # Portfolio tracking
        self.initial_capital = initial_capital
        self.cash = initial_capital
        
        # Price data cache for MTM calculations
        self.price_cache = {}  # Store price series for all traded assets
        self.all_trading_dates = set()  # All dates where we need MTM
        
        # Track all traded assets
        self.traded_assets = set()

        # Company index to ticker mapping
        self.index_cik_ticker_map = index_cik_ticker_map if index_cik_ticker_map is not None else {}

        # Configure logging within the class
        self.logger = logging.getLogger(self.__class__.__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
        # Cointegration monitoring state
        self.coint_retest_window_days = coint_retest_window_days
        self.coint_suspend_p = coint_suspend_p
        self.coint_reactivate_p = coint_reactivate_p
        self.coint_retest_freq = coint_retest_freq
        self.pair_states = {}  # (c1,c2) -> {'state': 'ACTIVE'|'SUSPENDED'|'WATCH','last_p': float,'fail_count': int}
        # Residual std summaries per pair for diagnostics
        self.residual_std_summary = {}
        # Cointegration test history
        self.coint_history = defaultdict(list)  # (c1,c2) -> [{'date':..., 'p':..., 'state':...}, ...]
        # Global monthly retest calendar computed in trade_pairs
        self.global_retest_days = set()
        # Guard-rail parameters
        self.min_price = min_price
        self.max_price = max_price
        self.resid_std_floor = resid_std_floor
        self.z_score_cap = z_score_cap
        self.max_abs_leg_return = max_abs_leg_return
        # WATCH state sizing
        self.watch_scale = watch_scale
        # Data coverage requirement
        self.require_full_tail = require_full_tail
        self.tail_date = pd.Timestamp(tail_date)
        # Skip / diagnostics tracking
        self.skip_reasons = defaultdict(int)

    def _should_retest(self, date: pd.Timestamp) -> bool:
        if self.coint_retest_freq == 'monthly':
            # first business day of month
            month_bd = pd.bdate_range(start=date.replace(day=1), end=date.replace(day=1) + pd.offsets.MonthEnd(0))
            return len(month_bd) and date == month_bd[0]
        return False

    def _run_coint_test(self, company1, company2, time_series_data, asof_date: pd.Timestamp):
        start = asof_date - pd.Timedelta(days=self.coint_retest_window_days*1.6)  # fetch wider then tail
        start_date = start.strftime('%Y-%m-%d')
        end_date = (asof_date - pd.Timedelta(days=1)).strftime('%Y-%m-%d')  # no lookahead
        s1 = self.get_price_series(company1, time_series_data, start_date=start_date, end_date=end_date)
        s2 = self.get_price_series(company2, time_series_data, start_date=start_date, end_date=end_date)
        if s1 is None or s2 is None:
            return None
        df = pd.DataFrame({'s1': s1, 's2': s2}).dropna()
        if len(df) > self.coint_retest_window_days:
            df = df.tail(self.coint_retest_window_days)
        if len(df) < COINT_MIN_POINTS or df['s1'].nunique() <= 1 or df['s2'].nunique() <= 1:
            return None
        y = df['s1']; X = sm.add_constant(df['s2'])
        try:
            model = sm.OLS(y, X, missing='drop').fit()
            resid = y - model.predict(X)
            adf_p = adfuller(resid.dropna())[1]
            return adf_p
        except Exception:
            return None

    def _update_pair_state(self, pair_key, p_value):
        st = self.pair_states.get(pair_key, {'state': 'ACTIVE', 'last_p': None, 'fail_count': 0, 'pass_count': 0})
        if p_value is None:
            self.pair_states[pair_key] = st
            return st['state']
        st['last_p'] = p_value
        # State transitions with WATCH
        if st['state'] == 'ACTIVE':
            if p_value >= max(self.coint_suspend_p, 0.20):
                # Hard fail: suspend immediately
                st['state'] = 'SUSPENDED'
                st['fail_count'] = 1
                st['pass_count'] = 0
            elif p_value >= self.coint_suspend_p:
                # Warning: go to WATCH
                st['state'] = 'WATCH'
                st['fail_count'] = 1
                st['pass_count'] = 0
            else:
                st['fail_count'] = 0
                st['pass_count'] += 1
        elif st['state'] == 'WATCH':
            if p_value >= max(self.coint_suspend_p, 0.20) or st.get('fail_count', 0) >= 1 and p_value >= self.coint_suspend_p:
                # Second fail or hard fail → suspend
                st['state'] = 'SUSPENDED'
                st['fail_count'] += 1
                st['pass_count'] = 0
            elif p_value < self.coint_reactivate_p:
                # Clean pass → back to ACTIVE
                st['state'] = 'ACTIVE'
                st['pass_count'] += 1
                st['fail_count'] = 0
            else:
                # Stay in WATCH
                st['pass_count'] = 0
        elif st['state'] == 'SUSPENDED':
            if p_value < self.coint_reactivate_p:
                st['pass_count'] = st.get('pass_count', 0) + 1
                st['fail_count'] = 0
                if st['pass_count'] >= 2:
                    st['state'] = 'ACTIVE'
            else:
                st['fail_count'] += 0  # remain suspended
        self.pair_states[pair_key] = st
        # Log history
        self.coint_history[pair_key].append({'date': None, 'p': p_value, 'state': st['state']})
        return st['state']

    def get_price_series(self, company_index, time_series_data, start_date=None, end_date=None):
        """
        Fetch the price series for a given company index, filtered by start and end dates.
        """
        if str(company_index) not in self.index_cik_ticker_map:
            return None
        tickers = self.index_cik_ticker_map[str(company_index)]["ticker"]

        for ticker in tickers:
            ticker_data = time_series_data[time_series_data['ticker'] == ticker]
            if not ticker_data.empty:
                timeseries = ticker_data.iloc[0]['timeseries']
                timeseries = pd.Series(timeseries)
                timeseries.index = pd.to_datetime(timeseries.index)

                # Filter by date range if specified
                if start_date and end_date:
                    timeseries = timeseries[start_date:end_date]
                return timeseries.sort_index()
        return None

    def get_price_on_date(self, company_index, date):
        """
        Get price for a company on a specific date, using last available price if date not available.
        """
        if company_index not in self.price_cache:
            return None
        
        price_series = self.price_cache[company_index]
        if date in price_series.index:
            return price_series.loc[date]
        else:
            # Use last available price before this date
            available_dates = price_series.index[price_series.index <= date]
            if len(available_dates) > 0:
                return price_series.loc[available_dates[-1]]
        return None

    def trade_pair(self, pair, time_series_data, start_date, end_date):
        """
        Trade a single pair over a specified period.

        Parameters:
        - pair: Dictionary with 'Company1' and 'Company2' IDs
        - time_series_data: DataFrame containing price data for all companies
        - start_date: Start date for trading
        - end_date: End date for trading
        """
        # Fetch price series with a warm-up buffer before OOS start to compute rolling stats without delay
        # Use business-day buffer sized to the larger of indicator lookback (252) and COINT retest window (e.g., 504)
        warmup_days = max(252, self.coint_retest_window_days)
        bday = pd.tseries.offsets.BDay(warmup_days)
        buffer_start = (pd.Timestamp(start_date) - bday).strftime('%Y-%m-%d')
        series1 = self.get_price_series(pair['Company1'], time_series_data, start_date=buffer_start, end_date=end_date)
        series2 = self.get_price_series(pair['Company2'], time_series_data, start_date=buffer_start, end_date=end_date)

        # Check if both series exist and cover the entire OOS period
        if series1 is None or series2 is None:
            self.skip_reasons['no_series'] += 1
            return
        # Optional strict data tail requirement (can suppress early trading)
        if self.require_full_tail:
            if pd.Timestamp(series1.index[-1]) < self.tail_date or pd.Timestamp(series2.index[-1]) < self.tail_date:
                self.skip_reasons['insufficient_tail'] += 1
                return

        # Store price series in cache for MTM calculations
        self.price_cache[pair['Company1']] = series1
        self.price_cache[pair['Company2']] = series2
        
        # Track traded assets
        self.traded_assets.add(pair['Company1'])
        self.traded_assets.add(pair['Company2'])

        # Align lengths and dates
        combined_df = pd.DataFrame({'series1': series1, 'series2': series2}).dropna()
        if len(combined_df) < 60:
            self.skip_reasons['short_overlap'] += 1
            return

        # Rolling, no-lookahead regression using rolling cov/var to estimate beta and alpha
        lookback = 252  # approx 1 trading year
        s1 = combined_df['series1']
        s2 = combined_df['series2']
        roll_mean_s1 = s1.rolling(lookback).mean()
        roll_mean_s2 = s2.rolling(lookback).mean()
        roll_var_s2 = s2.rolling(lookback).var()
        roll_cov = s1.rolling(lookback).cov(s2)
        beta = (roll_cov / roll_var_s2).shift(1)  # shift to prevent lookahead
        alpha = (roll_mean_s1 - beta * roll_mean_s2).shift(1)
        predicted = (alpha + beta * s2)
        residuals = s1 - predicted
        # Rolling z-score of residuals (no-lookahead)
        resid_mean = residuals.rolling(lookback).mean().shift(1)
        resid_std = residuals.rolling(lookback).std(ddof=0).shift(1)
        z_score = (residuals - resid_mean) / resid_std
        combined_df['z_score'] = z_score

        # Now restrict iteration to OOS dates (after indicators are prepared) and include the z_score column
        first_valid = combined_df['z_score'].first_valid_index()
        if first_valid is None:
            self.skip_reasons['no_zscore'] += 1
            return
        start_idx = max(pd.Timestamp(start_date), first_valid)
        loop_df = combined_df.loc[combined_df.index >= start_idx].copy()

        # Add trading dates to the set (only OOS dates)
        self.all_trading_dates.update(loop_df.index)

        # Record residual std diagnostics for this pair
        pair_key = (pair['Company1'], pair['Company2'])
        rs = resid_std.dropna()
        if len(rs) > 0:
            self.residual_std_summary[pair_key] = {
                'mean': float(rs.mean()),
                'median': float(rs.median()),
                'min': float(rs.min()),
                'max': float(rs.max()),
                'count': int(rs.count())
            }

        # Use global monthly retest calendar if available; fallback to local
        if getattr(self, 'global_retest_days', None):
            retest_days = self.global_retest_days
        else:
            retest_days = set()
            _seen_months = set()
            for d in combined_df.index:
                key = (d.year, d.month)
                if key not in _seen_months:
                    _seen_months.add(key)
                    retest_days.add(d)

        position = None
        last_z_score = None  # To track crossover direction

        prev_price1 = None
        prev_price2 = None
        for date, row in loop_df.iterrows():
            z_score = row['z_score']
            price1 = row['series1']
            price2 = row['series2']

            # Skip dates before indicators are ready
            if pd.isna(z_score):
                last_z_score = z_score
                continue

            # Entry and exit conditions
            pair_key = (pair['Company1'], pair['Company2'])
            # Periodic cointegration re-test
            if date in retest_days:
                p_val = self._run_coint_test(pair_key[0], pair_key[1], time_series_data, date)
                state = self._update_pair_state(pair_key, p_val)
                # backfill date in last history entry
                try:
                    if self.coint_history[pair_key]:
                        self.coint_history[pair_key][-1]['date'] = date
                except Exception:
                    pass
                if state == 'SUSPENDED' and position is not None:
                    # Force close at current prices
                    pnl = position.close(exit_date=date, exit_price1=price1, exit_price2=price2)
                    self.cumulative_pnl += pnl; self.total_trades += 1
                    self.positions.append(position); self.daily_pnl[date] += pnl; self.cash += pnl
                    if position in self.open_positions: self.open_positions.remove(position)
                    position = None
            else:
                state = self.pair_states.get(pair_key, {'state': 'ACTIVE'})['state']

            # Guard-rails for entries
            allow_entry = True
            if (price1 <= self.min_price) or (price2 <= self.min_price) or (price1 >= self.max_price) or (price2 >= self.max_price):
                allow_entry = False
            # Residual std floor
            rsd = resid_std.loc[date] if date in resid_std.index else np.nan
            if pd.notna(rsd) and rsd < self.resid_std_floor:
                allow_entry = False
            # Z-score cap
            if pd.notna(z_score) and abs(z_score) > self.z_score_cap:
                allow_entry = False
            # Leg return cap
            if prev_price1 is not None and prev_price2 is not None and prev_price1 != 0 and prev_price2 != 0:
                r1 = abs(price1 / prev_price1 - 1.0)
                r2 = abs(price2 / prev_price2 - 1.0)
                if (r1 > self.max_abs_leg_return) or (r2 > self.max_abs_leg_return):
                    allow_entry = False

            if position is None and state in ('ACTIVE','WATCH') and allow_entry:
                # Entry conditions with crossover
                if last_z_score is not None:
                    if last_z_score < self.delta and z_score > self.delta:
                        # Crossed above +δ: Short the spread
                        scale = 1.0 if state == 'ACTIVE' else self.watch_scale
                        units1 = -(self.capital_per_pair * scale / 2) / price1  # Short asset 1
                        units2 = (self.capital_per_pair * scale / 2) / price2   # Long asset 2
                        position = Position(pair=(pair['Company1'], pair['Company2']),
                                            entry_date=date, position_type='short_spread',
                                            units1=units1, units2=units2,
                                            entry_price1=price1, entry_price2=price2)
                        self.open_positions.append(position)
                        # Note: In dollar-neutral pairs trading, cash impact is minimal
                        # We're essentially borrowing asset1 and buying asset2
                        
                    elif last_z_score > -self.delta and z_score < -self.delta:
                        # Crossed below -δ: Long the spread
                        scale = 1.0 if state == 'ACTIVE' else self.watch_scale
                        units1 = (self.capital_per_pair * scale / 2) / price1    # Long asset 1
                        units2 = -(self.capital_per_pair * scale / 2) / price2   # Short asset 2
                        position = Position(pair=(pair['Company1'], pair['Company2']),
                                            entry_date=date, position_type='long_spread',
                                            units1=units1, units2=units2,
                                            entry_price1=price1, entry_price2=price2)
                        self.open_positions.append(position)
                        
            elif position is not None:
                # Exit conditions
                exit_signal = False
                if (position.position_type == 'short_spread' and z_score <= 0) or \
                   (position.position_type == 'long_spread' and z_score >= 0):
                    exit_signal = True
                    
                elif abs(z_score) >= 2 * self.delta:
                    # Stop-loss condition
                    exit_signal = True
                
                if exit_signal:
                    pnl = position.close(exit_date=date, exit_price1=price1, exit_price2=price2)
                    self.cumulative_pnl += pnl
                    self.total_trades += 1
                    self.positions.append(position)
                    
                    # Record daily realized P&L
                    self.daily_pnl[date] += pnl
                    
                    # Remove from open positions
                    if position in self.open_positions:
                        self.open_positions.remove(position)
                    
                    # Update cash with realized P&L
                    self.cash += pnl
                    
                    position = None

            last_z_score = z_score
            prev_price1, prev_price2 = price1, price2

    def force_liquidation(self, end_date):
        """Force-close all open positions at the given end_date prices."""
        for pos in list(self.open_positions):
            price1 = self.get_price_on_date(pos.pair[0], pd.Timestamp(end_date))
            price2 = self.get_price_on_date(pos.pair[1], pd.Timestamp(end_date))
            if price1 is not None and price2 is not None:
                pnl = pos.close(exit_date=pd.Timestamp(end_date), exit_price1=price1, exit_price2=price2)
                self.cumulative_pnl += pnl
                self.daily_pnl[pd.Timestamp(end_date)] += pnl
                self.cash += pnl
                self.positions.append(pos)
            if pos in self.open_positions:
                self.open_positions.remove(pos)

    def calculate_daily_mtm(self, time_series_data, start_date, end_date):
        """
        Calculate daily mark-to-market portfolio values after all trading is complete.
        
        This method should be called after trade_pairs() to compute MTM for all dates.
        """
        # Get all unique dates from start to end
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Filter to business days only (markets are closed on weekends)
        business_days = pd.bdate_range(start=start_date, end=end_date)
        
        # Combine with actual trading dates to ensure we have all relevant dates
        all_dates = sorted(set(business_days) | self.all_trading_dates)
        all_dates = [d for d in all_dates if start_date <= d <= end_date]
        
        # Create position timeline
        position_timeline = []
        for pos in self.positions + self.open_positions:
            position_timeline.append({
                'position': pos,
                'entry_date': pos.entry_date,
                'exit_date': pos.exit_date if pos.exit_date else pd.Timestamp(end_date)
            })
        
        # Calculate MTM for each date
        current_cash = self.initial_capital
        
        for date in tqdm(all_dates, desc="Calculating daily MTM"):
            # Update cash based on realized P&L up to this date
            realized_pnl_today = self.daily_pnl.get(date, 0.0)
            current_cash = self.initial_capital + sum([self.daily_pnl[d] for d in self.daily_pnl if d <= date])
            
            # Find all positions open on this date
            # Consider a position open on dates strictly before its exit_date (EOD accounting)
            open_on_date = [
                pt['position'] for pt in position_timeline
                if pt['entry_date'] <= date and pt['exit_date'] > date
            ]
            
            # Calculate unrealized P&L for all open positions
            unrealized_pnl = 0.0
            for pos in open_on_date:
                company1, company2 = pos.pair
                
                # Get prices for this date (use last available if not trading day)
                price1 = self.get_price_on_date(company1, date)
                price2 = self.get_price_on_date(company2, date)
                
                if price1 is not None and price2 is not None:
                    unrealized_pnl += pos.get_mtm_value(price1, price2)
            
            # Record daily values
            self.daily_unrealized_pnl[date] = unrealized_pnl
            self.daily_cash[date] = current_cash
            self.daily_portfolio_value[date] = current_cash + unrealized_pnl

    def trade_pairs(self, cointegrated_pairs, time_series_data, start_date, end_date):
        """
        Trade multiple cointegrated pairs over a specified period.

        Parameters:
        - cointegrated_pairs: List of pairs (each pair is a dictionary with 'Company1' and 'Company2')
        - time_series_data: DataFrame containing price data for all companies
        - start_date: Start date for trading
        - end_date: End date for trading
        """
        self.logger.info(f"Trading {len(cointegrated_pairs)} pairs from {start_date} to {end_date}")
        
        # Compute global monthly retest calendar (first business day of each month)
        bd = pd.bdate_range(start=start_date, end=end_date)
        self.global_retest_days = set()
        by_month = {}
        for d in bd:
            key = (d.year, d.month)
            if key not in by_month:
                by_month[key] = d
        self.global_retest_days = set(by_month.values())

        # Trade each pair
        for pair in tqdm(cointegrated_pairs, desc="Trading Cointegrated Pairs"):
            self.trade_pair(pair, time_series_data, start_date, end_date)
        # Force liquidation at end of period to convert unrealized to realized
        self.force_liquidation(end_date)

        # After all pairs are traded, calculate daily MTM portfolio values
        self.logger.info("Calculating daily mark-to-market values...")
        self.calculate_daily_mtm(time_series_data, pd.Timestamp(start_date), pd.Timestamp(end_date))

    def get_results(self, total_capital):
        """
        Retrieve the trading results with proper MTM calculations.

        Parameters:
        - total_capital: Total initial capital of the portfolio

        Returns:
        - Dictionary containing cumulative PnL, total trades, cumulative return, PnL series, and MTM portfolio value
        """
        try:
            if self.initial_capital != 0:
                cumulative_return = self.cumulative_pnl / self.initial_capital
            else:
                cumulative_return = 0
        except:
            cumulative_return = 0

        # Convert daily tracking dictionaries to pandas Series
        portfolio_value_series = pd.Series(self.daily_portfolio_value).sort_index()
        daily_pnl_series = pd.Series(self.daily_pnl).sort_index()
        cash_series = pd.Series(self.daily_cash).sort_index()
        unrealized_pnl_series = pd.Series(self.daily_unrealized_pnl).sort_index()
        
        # Final portfolio value
        if len(portfolio_value_series) > 0:
            final_portfolio_value = portfolio_value_series.iloc[-1]
        else:
            final_portfolio_value = self.initial_capital + self.cumulative_pnl

        return {
            'CumulativePnL': self.cumulative_pnl,
            'TotalTrades': self.total_trades,
            'CumulativeReturn': cumulative_return,
            'PnLSeries': daily_pnl_series,
            'portfolio_value_series': portfolio_value_series,
            'cash_series': cash_series,
            'unrealized_pnl_series': unrealized_pnl_series,
            'CumulativePortfolioValue': final_portfolio_value,
            'num_open_positions': len(self.open_positions)
        }

year = 2013
all_results = []
pnl_trajectories = {}
portfolio_trajectory = {}

# --------------------------------------------
# Main Execution
# --------------------------------------------

# Define your clusters (assuming these are already defined)
cluster_dfs = {
    'CD-Cluster': CD_cluster_df,
    'Rolling_CD_Cluster': Rolling_CD_cluster_df,
    'TPG-Cluster': TPG_Cluster_df,
    'SBERT-Cluster': SBERT_Cluster_df,
    'BERT-Cluster': BERT_Cluster_df,
    'SIC': year_SIC_cluster_df,
    'Industry': year_Industry_cluster_df
}

# clusters_to_process list means that these cluster groups haven't been ran, hence not saved yet (no info on these cointegration)
# We provide these cointegration data, but you can re-process it by uncommenting the below list:
clusters_to_process = ["CD-Cluster", "Rolling_CD_Cluster", "TPG-Cluster", 'SBERT-Cluster', 'BERT-Cluster', "SIC", "Industry"]

# We set this as an empty list since these files already exist in ./data/cointegration/, i.e ""./data/cointegration/cointegrated_pairs_CD-Cluster"
# This speeds up the process significantly.
clusters_to_process = ["CD-Cluster"]

clusters_to_process = []


os.makedirs("./data/cointegration/Traded Clusters/", exist_ok=True)
for cluster_type, cluster_df in cluster_dfs.items():
    print("\n", cluster_type, "\n")
    cointegrated_pairs_file = f'./data/cointegration/cointegrated_pairs_{cluster_type}.csv'
    if cluster_type in clusters_to_process:
        # Re-identify and save cointegrated pairs for this cluster
        try:
            cointegrated_pairs_df = identify_and_save_cointegrated_pairs(cluster_type, cluster_df, time_series_data, year)
            cointegrated_pairs = cointegrated_pairs_df.to_dict('records')
            logger.info(f"Re-identified and saved cointegrated pairs for {cluster_type}.")
        except Exception as e:
            logger.error(f"Error identifying and saving cointegrated pairs for {cluster_type}: {e}")
            continue
    else:
        # Load existing cointegrated pairs from file
        if os.path.exists(cointegrated_pairs_file):
            try:
                cointegrated_pairs_df = pd.read_csv(cointegrated_pairs_file)
                cointegrated_pairs = cointegrated_pairs_df.to_dict('records')
            except Exception as e:
                logger.error(f"Error loading {cointegrated_pairs_file}: {e}")
                continue
            print(f"Loaded cointegrated pairs for {cluster_type} from {cointegrated_pairs_file}")
        else:
            print(f"Cointegrated pairs file for {cluster_type} not found. Skipping.")
            continue

    # Proceed to trade the pairs using the PairTrader class
    # Filter pairs based on p-value and correlation (NO CAP; use all qualifying pairs)
    try:
        df_pairs = pd.DataFrame(cointegrated_pairs)
        if {'p-value', 'Correlation'}.issubset(df_pairs.columns):
            df_pairs = df_pairs[(df_pairs['p-value'] < 0.01) & (df_pairs['Correlation'] > 0.95)]
            # Sort by best statistics: lowest p-value, then highest correlation
            df_pairs = df_pairs.sort_values(by=['p-value', 'Correlation'], ascending=[True, False])
            cointegrated_pairs = df_pairs.to_dict('records')
        else:
            # Fallback if columns are missing
            cointegrated_pairs = [pair for pair in cointegrated_pairs if (pair.get('p-value', 1.0) < 0.01) and (pair.get('Correlation', 0.0) > 0.95)]
        pd.DataFrame(cointegrated_pairs).to_csv(f'./data/cointegration/Traded Clusters/NEW_cointegrated_pairs_{cluster_type}.csv', index=False)
    except Exception as e:
        logger.warning(f"Filtering/sorting pairs failed for {cluster_type} with error: {e}. Using original list (no cap).")
    print(f"Number of pairs to trade (no cap applied): {len(cointegrated_pairs)}")

    # Initialize PairTrader with MTM capabilities
    trader = PairTrader(capital_per_pair=10000, delta=1.0, initial_capital=100000, index_cik_ticker_map=index_cik_ticker_map)

    # Trade pairs (this now includes automatic MTM calculation)
    trader.trade_pairs(cointegrated_pairs, time_series_data, "2014-01-01", "2020-12-31")
    print(f"Traded: {trader.total_trades}, Open positions remaining: {len(trader.open_positions)}")
    logger.info(f"Total traded pairs: {trader.total_trades}")

    # Diagnostics wiring
    traders[cluster_type] = trader
    traded_pairs_by_cluster[cluster_type] = cointegrated_pairs
    residual_std_summary[cluster_type] = trader.residual_std_summary

    # Get results with proper MTM calculations
    total_capital = trader.capital_per_pair * len(cointegrated_pairs)
    results = trader.get_results(total_capital)
    # PV consistency check: final portfolio value should equal initial capital + realized PnL (after forced liquidation)
    try:
        expected_final = float(trader.initial_capital + trader.cumulative_pnl)
        observed_final = float(results.get('CumulativePortfolioValue', expected_final))
        if not np.isclose(observed_final, expected_final, rtol=0, atol=1e-6):
            logger.warning(f"PV consistency check FAILED for {cluster_type}: observed {observed_final:.2f} vs expected {expected_final:.2f}")
        else:
            logger.info(f"PV consistency check passed for {cluster_type}: {observed_final:.2f} == {expected_final:.2f}")
    except Exception as e:
        logger.warning(f"PV consistency check encountered an error for {cluster_type}: {e}")

    # Print and plot MTM line immediately for this cluster
    try:
        mtm_series = results.get('portfolio_value_series', pd.Series(dtype=float))
        if len(mtm_series) > 0:
            # Compact textual summary
            print(f"\n[MTM] {cluster_type}: start={mtm_series.index[0].date()} end={mtm_series.index[-1].date()} "
                  f"min=${mtm_series.min():,.2f} max=${mtm_series.max():,.2f} final=${mtm_series.iloc[-1]:,.2f}")
            # Quick preview (head and tail)
            preview = pd.concat([mtm_series.head(3), mtm_series.tail(3)])
            print(preview.to_string())
            # Quick per-cluster plot
            fig, ax = plt.subplots(1, 1, figsize=(10, 4))
            ax.plot(mtm_series.index, mtm_series.values, label=cluster_type, linewidth=2)
            ax.axhline(y=trader.initial_capital, color='gray', linestyle='--', alpha=0.5, label='Initial Capital')
            ax.set_title(f"MTM Portfolio Value - {cluster_type}")
            ax.set_xlabel("Date"); ax.set_ylabel("Portfolio Value ($)")
            ax.grid(True, alpha=0.3); ax.legend(loc='upper left')
            plt.tight_layout(); plt.show()
        else:
            print(f"[MTM] {cluster_type}: No MTM data available.")
    except Exception as e:
        print(f"Could not display MTM for {cluster_type}: {e}")
    
    # Calculate statistics
    number_of_cointegrated_pairs = 0
    try:
        number_of_cointegrated_pairs = len(cointegrated_pairs_df)
        number_of_total_pairs_in_this_cluster = 0
        # For the selection year only, sum nC2 over all clusters
        for cluster in cluster_df[cluster_df["year"].isin([year])]["clusters"]:
            for key, values in cluster.items():
                n = len(values)
                if n > 1:
                    number_of_total_pairs_in_this_cluster += n * (n - 1) // 2
    except Exception:
        pass

    percentage_of_cointegrated_pairs_to_total_number_of_pairs_in_this_cluster = 0
    try:
        percentage_of_cointegrated_pairs_to_total_number_of_pairs_in_this_cluster = number_of_cointegrated_pairs/number_of_total_pairs_in_this_cluster
    except:
        pass

    print("percentage_of_cointegrated_pairs_to_total_number_of_pairs_in_this_cluster:", percentage_of_cointegrated_pairs_to_total_number_of_pairs_in_this_cluster)

    all_results.append({
        'ClusterType': cluster_type,
        'PnL': results['CumulativePnL'],
        'Trades': results['TotalTrades'],
        'CumulativeReturn': results['CumulativeReturn'],
        'CumulativePortfolioValue': results['CumulativePortfolioValue'],
        'percentage_of_cointegrated_pairs': percentage_of_cointegrated_pairs_to_total_number_of_pairs_in_this_cluster,
        'OpenPositions': results['num_open_positions']
    })

    pnl_trajectories[cluster_type] = results['PnLSeries']
    portfolio_trajectory[cluster_type] = results['portfolio_value_series']

# Performance evaluation functions
def _pct_returns(value_series):
    """Calculate percentage returns from a value series"""
    v = value_series.sort_index()
    return v.pct_change().dropna()

def sharpe_ratio(value_series, rf_annual=0.0):
    """
    Calculate Sharpe ratio from portfolio value series
    
    Sharpe Ratio = (μ - rf) / σ * √252
    where μ is mean daily return, σ is std dev of daily returns
    """
    r = _pct_returns(value_series)
    if len(r) == 0:
        return np.nan
    excess = r - rf_annual / 252.0
    mu, sigma = excess.mean(), excess.std()
    return np.nan if sigma == 0 else (mu / sigma) * np.sqrt(252)

def max_drawdown(value_series):
    """Calculate maximum drawdown from portfolio value series"""
    if len(value_series) == 0:
        return np.nan
    cum_returns = value_series / value_series.iloc[0]
    running_max = cum_returns.cummax()
    drawdown = (cum_returns - running_max) / running_max
    return drawdown.min()

def calmar_ratio(value_series, rf_annual=0.0):
    """Calculate Calmar ratio (annualized return / max drawdown)"""
    if len(value_series) < 2:
        return np.nan
    total_return = (value_series.iloc[-1] / value_series.iloc[0]) - 1
    years = (value_series.index[-1] - value_series.index[0]).days / 365.25
    annual_return = (1 + total_return) ** (1/years) - 1 - rf_annual
    mdd = abs(max_drawdown(value_series))
    return annual_return / mdd if mdd != 0 else np.nan

def evaluate_portfolios(portfolio_trajectory, initial_capital=100000, rf_annual=0.0):
    """Evaluate portfolio performance metrics"""
    rows = []
    for name, series in portfolio_trajectory.items():
        if len(series) > 0:
            final_value = series.iloc[-1]
            total_return = (final_value / initial_capital - 1) * 100
            
            rows.append({
                "Cluster": name,
                "Sharpe": sharpe_ratio(series, rf_annual),
                "Max Drawdown": max_drawdown(series) * 100,  # As percentage
                "Calmar": calmar_ratio(series, rf_annual),
                "Final Portfolio Value": final_value,
                "Total Return (%)": total_return,
                "Annualized Return (%)": ((final_value/initial_capital) ** (252/len(series)) - 1) * 100
            })
    return pd.DataFrame(rows)

# Calculate and display metrics
print("\n=== PORTFOLIO PERFORMANCE METRICS ===")
metrics = evaluate_portfolios(portfolio_trajectory)
print(metrics.to_string(index=False))

# Display summary results
print("\n=== TRADING SUMMARY ===")
summary_df = pd.DataFrame(all_results)
print(summary_df.to_string(index=False))

# Plot the MTM portfolio values
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 1, figsize=(14, 10))

# Plot 1: Portfolio Value Over Time
ax1 = axes[0]
for cluster_type, series in portfolio_trajectory.items():
    if len(series) > 0:
        ax1.plot(series.index, series.values, label=cluster_type, linewidth=2)

ax1.axhline(y=100000, color='gray', linestyle='--', alpha=0.5, label='Initial Capital')
ax1.set_title("Mark-to-Market Portfolio Value Over Time", fontsize=14, fontweight='bold')
ax1.set_xlabel("Date")
ax1.set_ylabel("Portfolio Value ($)")
ax1.legend(loc='upper left')
ax1.grid(True, alpha=0.3)
ax1.set_ylim(bottom=0)

# Plot 2: Cumulative Returns
ax2 = axes[1]
for cluster_type, series in portfolio_trajectory.items():
    if len(series) > 0:
        returns = (series / 100000 - 1) * 100  # Percentage returns from initial
        ax2.plot(series.index, returns.values, label=cluster_type, linewidth=2)

ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax2.set_title("Cumulative Returns (%)", fontsize=14, fontweight='bold')
ax2.set_xlabel("Date")
ax2.set_ylabel("Return (%)")
ax2.legend(loc='upper left')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Print diagnostics
print("\n=== DIAGNOSTICS ===")
for cluster_type in portfolio_trajectory.keys():
    if cluster_type in portfolio_trajectory and len(portfolio_trajectory[cluster_type]) > 0:
        series = portfolio_trajectory[cluster_type]
        print(f"\n{cluster_type}:")
        print(f"  - Number of MTM data points: {len(series)}")
        print(f"  - Date range: {series.index[0].date()} to {series.index[-1].date()}")
        print(f"  - Min value: ${series.min():,.2f}")
        print(f"  - Max value: ${series.max():,.2f}")
        print(f"  - Final value: ${series.iloc[-1]:,.2f}")
