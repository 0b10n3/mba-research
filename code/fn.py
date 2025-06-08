import pandas as pd
import numpy as np
import yfinance as yf
from logger import logger
from datetime import datetime


def read_etf_list(path = "data/etfsListados.csv"):
    """
    Read the etfs list from a csv file.
    :param path:
    :return:
    """

    try:

        df = pd.read_csv(path, sep=";")

        if "Código" not in df.columns:
            print(f"Warning: Column 'Código' not found in {path}. Returning original DataFrame.")
            return df

        df["Código"] = df["Código"].astype(str) + "11.SA"

        return df

    except FileNotFoundError:
        print(f"Error: The file {path} was not found.")
        return pd.DataFrame()
    except Exception as e:
        print(f"An error occurred: {e}")
        return pd.DataFrame()


def get_yahoo_prices(tickers, start_date, end_date):
    """
    Get yahoo prices.
    :param tickers:
    :param start_date:
    :param end_date:
    :return:
    """

    price_data = pd.DataFrame()

    try:
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        all_data = yf.download(tickers,
                               start=start_dt,
                               end=end_dt,
                               progress=False,
                               auto_adjust=False)

        if all_data.empty:
            logger.error(f"No data found for {tickers}.")
            return price_data

        if isinstance(all_data.columns, pd.MultiIndex):
            if 'Adj Close' in all_data.columns.levels[0]:
                price_data = all_data['Adj Close']
            elif 'Close' in all_data.columns.levels[0]:
                price_data = all_data['Close']
                logger.info(
                    "Using adjusted close for {tickers} from {start_date} to {end_date}.")
            else:
                logger.error(
                    "No price data found for {tickers} from {start_date} to {end_date}.")
                return pd.DataFrame()
        elif isinstance(all_data, pd.DataFrame):
            if 'Adj Close' in all_data.columns:
                price_data = all_data[['Adj Close']]
            elif 'Close' in all_data.columns:
                price_data = all_data[['Close']]
                if len(tickers) == 1: logger.info(
                    f"Using adjusted close for {tickers} from {start_date} to {end_date}.")
            else:
                logger.error(
                    f"No price data found for {tickers} from {start_date} to {end_date}.")
                return pd.DataFrame()

            if len(tickers) == 1 and not price_data.empty:
                price_data.columns = tickers
        else:
            logger.error(f"Non expected data type {type(all_data)}.")
            return pd.DataFrame()

        price_data = price_data.dropna()
        if price_data.empty:
            logger.warning(
                f"No price data found for {tickers} from {start_date} to {end_date}.")
            return pd.DataFrame()
        return price_data
    except Exception as e:
        logger.error(f"Error getting prices for {tickers} from {start_date} to {end_date}.")
        return pd.DataFrame()


def get_stock_date_range(tickers):
    """
    Get the minimum and maximum available dates for historical stock data from Yahoo Finance.

    :param tickers: A string or a list of strings representing the stock ticker(s).
    :return: A pandas DataFrame with tickers as index and 'Min Date' and 'Max Date' as columns.
             Returns an empty DataFrame if no data is found or an error occurs.
    """
    if isinstance(tickers, str):
        tickers = [tickers]
    elif not isinstance(tickers, list):
        logger.error("Tickers must be a string or a list of strings.")
        return pd.DataFrame()

    date_ranges = {}

    for ticker_symbol in tickers:
        try:
            ticker_obj = yf.Ticker(ticker_symbol)

            hist = ticker_obj.history(period="max", auto_adjust=False)

            if hist.empty:
                logger.warning(f"No historical data found for {ticker_symbol}.")
                date_ranges[ticker_symbol] = {'Min Date': pd.NaT, 'Max Date': pd.NaT}
                continue

            min_date = hist.index.min()
            max_date = hist.index.max()

            if isinstance(min_date, pd.Timestamp) and min_date.tzinfo is not None:
                min_date = min_date.tz_localize(None)
            if isinstance(max_date, pd.Timestamp) and max_date.tzinfo is not None:
                max_date = max_date.tz_localize(None)


            date_ranges[ticker_symbol] = {'Min Date': min_date, 'Max Date': max_date}
            logger.info(f"Date range for {ticker_symbol}: Min Date - {min_date.strftime('%Y-%m-%d') if pd.notna(min_date) else 'N/A'}, Max Date - {max_date.strftime('%Y-%m-%d') if pd.notna(max_date) else 'N/A'}")

        except Exception as e:
            logger.error(f"Error fetching data for {ticker_symbol}: {e}")
            date_ranges[ticker_symbol] = {'Min Date': pd.NaT, 'Max Date': pd.NaT}

    if not date_ranges:
        return pd.DataFrame()

    result_df = pd.DataFrame.from_dict(date_ranges, orient='index')
    return result_df


def add_stock_date_range_to_df(etfs_df: pd.DataFrame, ticker_column_name: str = "Código"):
    """
    Fetches the minimum and maximum available historical data dates from Yahoo Finance
    for each ticker in the provided DataFrame and adds them as new columns.

    :param etfs_df: pandas DataFrame containing a column with stock tickers.
    :param ticker_column_name: The name of the column in etfs_df that holds the ticker symbols.
    :return: pandas DataFrame with two new columns 'Min Date' and 'Max Date' added.
             Returns the original DataFrame with NaT in new columns if errors occur or tickers are invalid.
    """
    if not isinstance(etfs_df, pd.DataFrame):
        logger.error("Input must be a pandas DataFrame.")
        return pd.DataFrame()

    if ticker_column_name not in etfs_df.columns:
        logger.error(f"Ticker column '{ticker_column_name}' not found in the DataFrame.")
        etfs_df['Min Date'] = pd.NaT
        etfs_df['Max Date'] = pd.NaT
        return etfs_df

    if etfs_df.empty:
        logger.info("Input DataFrame is empty. Adding empty 'Min Date' and 'Max Date' columns.")
        etfs_df['Min Date'] = pd.NaT
        etfs_df['Max Date'] = pd.NaT
        return etfs_df

    min_dates = []
    max_dates = []

    for ticker_symbol in etfs_df[ticker_column_name]:
        if not isinstance(ticker_symbol, str) or not ticker_symbol.strip():
            logger.warning(f"Invalid or empty ticker symbol encountered: {ticker_symbol}. Skipping.")
            min_dates.append(pd.NaT)
            max_dates.append(pd.NaT)
            continue
        try:
            ticker_obj = yf.Ticker(ticker_symbol)
            hist = ticker_obj.history(period="max", auto_adjust=False)

            if hist.empty:
                logger.warning(f"No historical data found for {ticker_symbol}.")
                min_dates.append(pd.NaT)
                max_dates.append(pd.NaT)
            else:
                min_date_val = hist.index.min()
                max_date_val = hist.index.max()

                if isinstance(min_date_val, pd.Timestamp) and min_date_val.tzinfo is not None:
                    min_date_val = min_date_val.tz_localize(None)
                if isinstance(max_date_val, pd.Timestamp) and max_date_val.tzinfo is not None:
                    max_date_val = max_date_val.tz_localize(None)

                min_dates.append(min_date_val)
                max_dates.append(max_date_val)
                logger.info(f"Date range for {ticker_symbol}: Min Date - {min_date_val.strftime('%Y-%m-%d') if pd.notna(min_date_val) else 'N/A'}, Max Date - {max_date_val.strftime('%Y-%m-%d') if pd.notna(max_date_val) else 'N/A'}")
        except Exception as e:
            logger.error(f"Error fetching data for {ticker_symbol}: {e}")
            min_dates.append(pd.NaT)
            max_dates.append(pd.NaT)

    etfs_df['Min Date'] = min_dates
    etfs_df['Max Date'] = max_dates

    return etfs_df


def get_tickers_before_date(
    df_with_dates: pd.DataFrame,
    cutoff_year: int = 2014,
    ticker_column_name: str = "Código",
    min_date_column_name: str = "Min Date"
) -> list:
    """
    Identifies tickers from a DataFrame that have historical data available
    prior to the beginning of a specified year.

    :param df_with_dates: pandas DataFrame, expected to be the output from
                          add_stock_date_range_to_df. It should contain
                          a ticker column and a 'Min Date' column.
    :param cutoff_year: int, the year before which data should be available.
                        The cutoff date will be January 1st of this year.
                        Defaults to 2014.
    :param ticker_column_name: str, the name of the column in df_with_dates
                               that holds the ticker symbols. Defaults to "Código".
    :param min_date_column_name: str, the name of the column in df_with_dates
                                 that holds the minimum available date for each ticker.
                                 Defaults to "Min Date".
    :return: list, a list of ticker symbols that have data prior to
             the specified cutoff_year. Returns an empty list if no such tickers
             are found or if input is invalid.
    """
    if not isinstance(df_with_dates, pd.DataFrame):
        print("Error: Input must be a pandas DataFrame.")
        return []

    if ticker_column_name not in df_with_dates.columns:
        print(f"Error: Ticker column '{ticker_column_name}' not found in the DataFrame.")
        return []

    if min_date_column_name not in df_with_dates.columns:
        print(f"Error: Min date column '{min_date_column_name}' not found in the DataFrame.")
        return []

    if df_with_dates.empty:
        return []

    cutoff_date = pd.Timestamp(datetime(cutoff_year, 1, 1))

    try:
        min_dates_series = pd.to_datetime(df_with_dates[min_date_column_name], errors='coerce')
    except Exception as e:
        print(f"Error converting '{min_date_column_name}' to datetime: {e}")
        return []

    filtered_df = df_with_dates[
        pd.notna(min_dates_series) & (min_dates_series < cutoff_date)
    ]

    if filtered_df.empty:
        return []

    tickers_list = filtered_df[ticker_column_name].tolist()

    return tickers_list


def build_assets_universe(cutoff_year: int = 2018):
    """
    Builds a universe of assets available for trading.
    :param cutoff_year:
    :return:
    """

    listed_etfs = read_etf_list()
    listed_etfs = add_stock_date_range_to_df(listed_etfs)
    return get_tickers_before_date(listed_etfs, cutoff_year=cutoff_year)


def build_portfolio(
        strategy_function,
        ref_month_str: str,
        asset_universe: list,
        all_prices_df: pd.DataFrame,
        lookback_years: int = 1,
        min_data_points_ratio: float = 0.90,
        returns_frequency: str = 'D',  # 'D' for daily, 'M' for monthly
        **strategy_kwargs
):
    """
    Builds a portfolio for a given reference month using a specified strategy function.

    The portfolio is constructed based on historical data available up to the end of
    the month *preceding* the ref_month_str.

    :param strategy_function: A function that accepts a DataFrame of asset returns
                              (and other **strategy_kwargs) and returns a dictionary
                              of asset tickers and their corresponding weights.
                              Example: {'TICKER1': 0.6, 'TICKER2': 0.4}
    :param ref_month_str: The reference month for which the portfolio is being constructed
                          (format: "YYYY-MM"). The portfolio is intended to be held
                          starting from this month.
    :param asset_universe: A list of ticker symbols considered available for inclusion
                           in the portfolio for the given ref_month.
    :param all_prices_df: A pandas DataFrame containing historical price data.
                          The index must be a DatetimeIndex, and columns should be
                          asset tickers.
    :param lookback_years: The number of years of historical data to retrieve prior to
                           the ref_month for strategy calculation.
    :param min_data_points_ratio: The minimum ratio of non-NaN data points required for
                                  an asset over the lookback period to be included in
                                  the strategy calculation.
    :param returns_frequency: The frequency for calculating returns. 'D' for daily,
                              'M' for monthly.
    :param strategy_kwargs: Additional keyword arguments to be passed directly to the
                            strategy_function (e.g., risk_free_rate for Markowitz).
    :return: A dictionary representing the portfolio, with asset tickers as keys
             and their weights as values. Returns an empty dictionary if a
             portfolio cannot be constructed.
    """
    logger.info(f"Initiating portfolio construction for reference month: {ref_month_str}.")
    logger.debug(
        f"Parameters: lookback_years={lookback_years}, "
        f"min_data_points_ratio={min_data_points_ratio}, "
        f"returns_frequency='{returns_frequency}', "
        f"strategy_kwargs={strategy_kwargs}")

    if not callable(strategy_function):
        logger.error("Provided 'strategy_function' is not callable.")
        return {}
    if not isinstance(asset_universe, list):
        logger.error("Provided 'asset_universe' is not a list.")
        return {}
    if not isinstance(all_prices_df, pd.DataFrame):
        logger.error("Provided 'all_prices_df' is not a pandas DataFrame.")
        return {}

    if not asset_universe:
        logger.warning("Asset universe is empty. Cannot build portfolio.")
        return {}
    if all_prices_df.empty:
        logger.warning("Historical prices DataFrame (all_prices_df) is empty. Cannot build portfolio.")
        return {}

    if not isinstance(all_prices_df.index, pd.DatetimeIndex):
        try:
            all_prices_df.index = pd.to_datetime(all_prices_df.index)
            logger.debug("Converted all_prices_df.index to DatetimeIndex.")
        except Exception as e:
            logger.error(f"Failed to convert all_prices_df.index to DatetimeIndex: {e}")
            return {}

    if all_prices_df.index.tz is not None:
        logger.debug(f"Original all_prices_df.index timezone: {all_prices_df.index.tz}. Converting to naive UTC.")
        all_prices_df.index = all_prices_df.index.tz_convert('UTC').tz_localize(None)

    try:
        first_day_of_ref_month = pd.to_datetime(f"{ref_month_str}-01")
        hist_data_end_date = first_day_of_ref_month - pd.Timedelta(days=1)
        hist_data_start_date = hist_data_end_date - pd.DateOffset(years=lookback_years) + pd.Timedelta(
            days=1)

        logger.info(
            f"Historical data window for strategy inputs: {hist_data_start_date.strftime('%Y-%m-%d')} to {hist_data_end_date.strftime('%Y-%m-%d')}.")

        available_tickers_in_prices = [ticker for ticker in asset_universe if ticker in all_prices_df.columns]
        if not available_tickers_in_prices:
            logger.warning(
                f"None of the {len(asset_universe)} assets in the universe are present as columns in all_prices_df. Example universe assets: {asset_universe[:3]}")
            return {}

        prices_window_df = all_prices_df.loc[
            (all_prices_df.index >= hist_data_start_date) &
            (all_prices_df.index <= hist_data_end_date),
            available_tickers_in_prices
        ]

        if prices_window_df.empty:
            logger.warning(
                f"No price data found within the lookback window for the selected assets. Window: {hist_data_start_date.strftime('%Y-%m-%d')} to {hist_data_end_date.strftime('%Y-%m-%d')}. Assets searched (max 3): {available_tickers_in_prices[:3]}.")
            return {}

        if returns_frequency.upper() == 'M':
            asset_returns_df = prices_window_df.resample('M').last().pct_change().dropna(how='all')
        elif returns_frequency.upper() == 'D':
            asset_returns_df = prices_window_df.pct_change().dropna(how='all')
        else:
            logger.error(f"Invalid 'returns_frequency': {returns_frequency}. Supported values are 'D' or 'M'.")
            return {}

        if asset_returns_df.empty:
            logger.warning(
                f"Asset returns DataFrame is empty after calculation (frequency: '{returns_frequency}'). Price window had {len(prices_window_df)} rows.")
            return {}

        if not asset_returns_df.index.empty:
            num_periods_in_returns = len(asset_returns_df.index)
        else:
            num_periods_in_returns = 0

        min_required_points = num_periods_in_returns * min_data_points_ratio

        final_asset_selection_for_strategy = []
        for ticker in asset_returns_df.columns:
            valid_data_points = asset_returns_df[ticker].count()
            if valid_data_points >= min_required_points:
                final_asset_selection_for_strategy.append(ticker)
            else:
                logger.info(
                    f"Ticker '{ticker}' excluded: Insufficient return data ({valid_data_points} actual vs {min_required_points:.1f} required based on {num_periods_in_returns} periods and ratio {min_data_points_ratio}).")

        if not final_asset_selection_for_strategy:
            logger.warning(
                "No assets passed the data availability filter (min_data_points_ratio). Cannot proceed with strategy.")
            return {}

        final_returns_df = asset_returns_df[final_asset_selection_for_strategy]

        if final_returns_df.empty:
            logger.warning("Final returns DataFrame for strategy is unexpectedly empty after asset selection.")
            return {}

        logger.info(
            f"Shape of final returns DataFrame for strategy: {final_returns_df.shape}. Assets (max 5): {final_returns_df.columns.tolist()[:5]}")

        logger.debug(f"Calling strategy function with {len(final_returns_df.columns)} assets.")
        portfolio_weights = strategy_function(final_returns_df, **strategy_kwargs)

        if not isinstance(portfolio_weights, dict):
            logger.error(
                f"The strategy_function did not return a dictionary (returned type: {type(portfolio_weights)}). Portfolio construction failed.")
            return {}

        processed_portfolio = {}
        for asset, weight in portfolio_weights.items():
            if isinstance(weight, (int, float)) and not np.isnan(weight) and abs(weight) > 1e-9:
                if asset not in final_returns_df.columns:
                    logger.warning(
                        f"Strategy returned weight for asset '{asset}' which was not in the final returns data provided to it. Ignoring this asset.")
                    continue
                processed_portfolio[asset] = weight
            else:
                logger.info(
                    f"Weight for asset '{asset}' ({weight}) is invalid or zero, excluding from final portfolio.")

        sum_weights = sum(processed_portfolio.values())
        if not (abs(sum_weights - 1.0) < 1e-4 or abs(sum_weights) < 1e-9):
            logger.warning(
                f"Final portfolio weights sum to {sum_weights:.6f}, which is not close to 1.0 (or 0.0 for cash). This may be intended by the strategy or indicate an issue.")

        if not processed_portfolio:
            logger.warning("Strategy resulted in an empty portfolio after processing weights.")
            return {}

        logger.info(f"Successfully constructed portfolio for reference month {ref_month_str}: {processed_portfolio}")
        return processed_portfolio

    except Exception as e:
        logger.error(f"An unexpected error occurred in build_portfolio for {ref_month_str}: {e}",
                     exc_info=True)
        return {}