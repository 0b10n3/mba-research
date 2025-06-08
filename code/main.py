from fn import build_assets_universe, get_yahoo_prices, build_portfolio
from strategies import markowitz_mean_variance_optimization, one_over_n_portfolio_strategy


etfs_before_2018 = build_assets_universe(cutoff_year=2018)

prices = get_yahoo_prices(etfs_before_2018, start_date='2018-01-01', end_date='2024-12-31')

portfolio_MVT = build_portfolio(
    strategy_function=markowitz_mean_variance_optimization,
    ref_month_str="2019-01",
    asset_universe=etfs_before_2018,
    all_prices_df=prices
)


portfolio_One_N = build_portfolio(
    strategy_function=one_over_n_portfolio_strategy,
    ref_month_str="2019-01",
    asset_universe=etfs_before_2018,
    all_prices_df=prices
)
