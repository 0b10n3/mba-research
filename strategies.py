from logger import logger

import numpy as np
import pandas as pd
from scipy.optimize import minimize


def _calculate_portfolio_stats(weights: np.ndarray,
                               expected_returns_annualized: pd.Series,
                               cov_matrix_annualized: pd.DataFrame):
    """
    Helper function to calculate annualized portfolio return and volatility.
    """

    weights_1d = np.array(weights).flatten()
    portfolio_return_annualized = np.sum(weights_1d * expected_returns_annualized)

    portfolio_variance_annualized = np.dot(weights_1d.T, np.dot(cov_matrix_annualized, weights_1d))


    if portfolio_variance_annualized < 0 and np.isclose(portfolio_variance_annualized, 0):
        portfolio_variance_annualized = 0.0
    elif portfolio_variance_annualized < 0:
        logger.warning(
            f"Calculated negative portfolio variance: {portfolio_variance_annualized:.8e}. "
            "This may indicate issues with the covariance matrix or extreme weights. Clamping to 0."
        )
        portfolio_variance_annualized = 0.0

    portfolio_volatility_annualized = np.sqrt(portfolio_variance_annualized)
    return portfolio_return_annualized, portfolio_volatility_annualized


def markowitz_mean_variance_optimization(
        returns_df: pd.DataFrame,
        risk_free_rate: float = 0.0,  # Annualized
        returns_frequency: str = 'D',  # 'D' for daily, 'M' for monthly
        optimization_objective: str = "maximize_sharpe",  # "minimize_volatility", "maximize_return"
        target_return: float = None,  # Annualized, required for "minimize_volatility"
        target_volatility: float = None,  # Annualized, required for "maximize_return"
        allow_short_selling: bool = False,
        weight_bounds_per_asset: tuple = (0.0, 1.0)  # (min_weight, max_weight) for each asset
):
    """
    Implements Markowitz Mean-Variance Portfolio Optimization.

    :param returns_df: DataFrame of historical asset returns (e.g., daily or monthly).
                       Index should be datetime, columns are asset tickers.
    :param risk_free_rate: Annualized risk-free rate.
    :param returns_frequency: Frequency of input returns_df ('D' for daily, 'M' for monthly).
                              This determines the annualization factor.
    :param optimization_objective: The goal of the optimization.
                                   - "maximize_sharpe": Maximizes the Sharpe Ratio.
                                   - "minimize_volatility": Minimizes portfolio volatility for a given target_return.
                                   - "maximize_return": Maximizes portfolio return for a given target_volatility.
    :param target_return: Annualized target return, required if objective is "minimize_volatility".
    :param target_volatility: Annualized target volatility, required if objective is "maximize_return".
    :param allow_short_selling: Boolean. If False, weights are constrained to be non-negative (lower bound of
                                weight_bounds_per_asset will be clamped at 0). If True, negative weights
                                are allowed as per weight_bounds_per_asset.
    :param weight_bounds_per_asset: Tuple (min_weight, max_weight) for each asset.
                                    E.g., (0.0, 1.0) for no short selling, max 100% per asset.
                                    E.g., (-0.5, 1.0) if short selling up to 50% is allowed.
    :return: Dictionary of asset tickers and their optimal weights.
             Returns an empty dictionary if optimization fails or inputs are invalid.
    """


    logger.info(
        f"Starting Markowitz optimization. Objective: '{optimization_objective}', Assets: {returns_df.shape[1]}, Returns Freq: '{returns_frequency}'.")

    num_assets = returns_df.shape[1]
    if num_assets == 0:
        logger.warning("Markowitz: Input returns_df contains no assets (columns). Returning empty portfolio.")
        return {}

    tickers = returns_df.columns.tolist()

    if returns_frequency.upper() == 'D':
        annualization_factor = 252
    elif returns_frequency.upper() == 'M':
        annualization_factor = 12
    else:
        logger.warning(
            f"Markowitz: Unknown returns_frequency '{returns_frequency}'. Defaulting to annualization_factor=1 (no annualization).")
        annualization_factor = 1

    mean_period_returns = returns_df.mean()
    expected_returns_annualized = mean_period_returns * annualization_factor

    cov_matrix_period = returns_df.cov()
    cov_matrix_annualized = cov_matrix_period * annualization_factor

    min_b, max_b = weight_bounds_per_asset
    if not allow_short_selling:
        min_b = max(0.0, min_b)

    if min_b > max_b:
        logger.error(
            f"Markowitz: Invalid weight_bounds_per_asset ({min_b}, {max_b}). Min bound cannot be greater than Max bound.")
        return {}

    if num_assets > 0 and min_b * num_assets > 1.0001:
            f"Markowitz: Sum of minimum weights ({min_b * num_assets:.2f}) > 1. This may lead to an infeasible solution.")
    if num_assets > 0 and max_b * num_assets < 0.9999 and not (num_assets == 1 and np.isclose(max_b, 1.0)):
        logger.warning(
            f"Markowitz: Sum of maximum weights ({max_b * num_assets:.2f}) < 1. This may lead to an infeasible solution if full investment is required.")

    bounds_per_weight = tuple((min_b, max_b) for _ in range(num_assets))
    initial_weights = np.array([1.0 / num_assets] * num_assets)

    constraints_base = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1.0})

    optimal_weights_array = None
    optimization_status_message = "Optimization not run."

    optimizer_options = {'ftol': 1e-9, 'maxiter': 1000, 'disp': False}

    if optimization_objective.lower() == "maximize_sharpe":
        def neg_sharpe_ratio_objective(weights):
            portfolio_return, portfolio_volatility = _calculate_portfolio_stats(weights, expected_returns_annualized,
                                                                                cov_matrix_annualized)
            if np.isclose(portfolio_volatility, 0.0):
                if np.isclose(portfolio_return - risk_free_rate,
                              0.0): return 0.0
                return -np.inf if portfolio_return > risk_free_rate else np.inf
            return -(portfolio_return - risk_free_rate) / portfolio_volatility

        result = minimize(neg_sharpe_ratio_objective, initial_weights, method='SLSQP',
                          bounds=bounds_per_weight, constraints=constraints_base, options=optimizer_options)
        optimal_weights_array = result.x
        optimization_status_message = result.message
        if not result.success: logger.warning(f"Markowitz (Maximize Sharpe) optimization issues: {result.message}")


    elif optimization_objective.lower() == "minimize_volatility":
        if target_return is None:
            logger.error("Markowitz: 'target_return' must be provided for 'minimize_volatility' objective.")
            return {}

        def portfolio_volatility_objective(weights):
            _, portfolio_volatility = _calculate_portfolio_stats(weights, expected_returns_annualized,
                                                                 cov_matrix_annualized)
            return portfolio_volatility

        current_constraints = [
            constraints_base,
            {'type': 'eq', 'fun': lambda weights:
            _calculate_portfolio_stats(weights, expected_returns_annualized, cov_matrix_annualized)[0] - target_return}
        ]
        result = minimize(portfolio_volatility_objective, initial_weights, method='SLSQP',
                          bounds=bounds_per_weight, constraints=current_constraints, options=optimizer_options)
        optimal_weights_array = result.x
        optimization_status_message = result.message
        if not result.success: logger.warning(f"Markowitz (Minimize Volatility) optimization issues: {result.message}")

    elif optimization_objective.lower() == "maximize_return":
        if target_volatility is None:
            logger.error("Markowitz: 'target_volatility' must be provided for 'maximize_return' objective.")
            return {}

        def neg_portfolio_return_objective(weights):
            portfolio_return, _ = _calculate_portfolio_stats(weights, expected_returns_annualized,
                                                             cov_matrix_annualized)
            return -portfolio_return

        current_constraints = [
            constraints_base,
            {'type': 'ineq', 'fun': lambda weights: target_volatility -
                                                    _calculate_portfolio_stats(weights, expected_returns_annualized,
                                                                               cov_matrix_annualized)[1]}
        ]
        result = minimize(neg_portfolio_return_objective, initial_weights, method='SLSQP',
                          bounds=bounds_per_weight, constraints=current_constraints, options=optimizer_options)
        optimal_weights_array = result.x
        optimization_status_message = result.message
        if not result.success: logger.warning(f"Markowitz (Maximize Return) optimization issues: {result.message}")
    else:
        logger.error(f"Markowitz: Unknown 'optimization_objective': '{optimization_objective}'.")
        return {}

    if optimal_weights_array is None:
        logger.error(
            f"Markowitz: optimal_weights_array is None after optimization attempt for '{optimization_objective}'. Status: {optimization_status_message}")
        return {}

    if not np.isclose(np.sum(optimal_weights_array), 1.0, atol=1e-4):
        logger.error(
            f"Markowitz: Optimal weights do not sum to 1 (sum: {np.sum(optimal_weights_array):.6f}). Optimization likely failed critically. Status: {optimization_status_message}")
        return {}

    optimal_weights_array[np.abs(optimal_weights_array) < 1e-6] = 0.0

    current_sum_after_cleaning = np.sum(optimal_weights_array)
    if not np.isclose(current_sum_after_cleaning, 0.0) and \
            np.isclose(current_sum_after_cleaning, 1.0, atol=1e-4) and \
            not np.isclose(current_sum_after_cleaning, 1.0):
        optimal_weights_array = optimal_weights_array / current_sum_after_cleaning
        logger.debug("Re-normalized weights slightly after cleaning tiny values.")

    final_portfolio_dict = {
        ticker: weight
        for ticker, weight in zip(tickers, optimal_weights_array)
        if abs(weight) > 1e-7
    }

    if not final_portfolio_dict:
        logger.warning("Markowitz: Optimization resulted in an empty portfolio (all weights are effectively zero).")
        return {}

    final_weights_for_stats = np.array([final_portfolio_dict.get(ticker, 0.0) for ticker in tickers])
    est_return, est_volatility = _calculate_portfolio_stats(final_weights_for_stats, expected_returns_annualized,
                                                            cov_matrix_annualized)
    est_sharpe = (est_return - risk_free_rate) / est_volatility if not np.isclose(est_volatility, 0) else np.nan

    logger.info(
        f"Markowitz Optimized Portfolio - Objective: {optimization_objective} (Status: {optimization_status_message})")
    logger.info(
        f"  Est. Annual Return: {est_return:.4%}, Est. Annual Volatility: {est_volatility:.4%}, Est. Sharpe Ratio: {est_sharpe:.4f}")
    logger.debug(f"  Final Portfolio Weights: {final_portfolio_dict}")

    return final_portfolio_dict


def one_over_n_portfolio_strategy(
        returns_df: pd.DataFrame,
        max_assets_cap: int = None
) -> dict:
    """
    Implements the 1/N (Equal Weight) portfolio construction strategy.

    This strategy allocates capital equally among the available assets. The actual
    return values in returns_df are not used for weight calculation, only the
    list of assets derived from its columns (which are assumed to be pre-filtered
    and available for investment).

    :param returns_df: pandas DataFrame of historical asset returns. The columns
                       of this DataFrame are used to identify the available assets.
    :param max_assets_cap: Optional integer. If provided and positive, the portfolio
                           will include at most this many assets. Assets are selected
                           based on their order in `returns_df.columns`. If None, or if
                           the cap is greater than or equal to the number of available
                           assets, all assets in `returns_df` are used.
    :return: A dictionary where keys are asset tickers (strings) and values are
             their corresponding equal weights (floats). Returns an empty
             dictionary if no assets are available or selected.
    """

    strategy_name = "1/N Portfolio Strategy"
    logger.info(
        f"{strategy_name}: Initiating. Original number of available assets from returns_df: {returns_df.shape[1]}.")

    if not isinstance(returns_df, pd.DataFrame):
        logger.error(
            f"{strategy_name}: Input 'returns_df' is not a pandas DataFrame. Received type: {type(returns_df)}.")
        return {}

    asset_tickers = returns_df.columns.tolist()
    num_available_assets = len(asset_tickers)

    if num_available_assets == 0:
        logger.warning(f"{strategy_name}: Input returns_df has no asset columns. Returning empty portfolio.")
        return {}

    selected_tickers_for_portfolio = asset_tickers

    if max_assets_cap is not None:
        if not isinstance(max_assets_cap, int):
            logger.warning(
                f"{strategy_name}: 'max_assets_cap' parameter is not an integer (received: {max_assets_cap}, type: {type(max_assets_cap)}). Ignoring cap.")
        elif max_assets_cap <= 0:
            logger.warning(
                f"{strategy_name}: 'max_assets_cap' ({max_assets_cap}) is non-positive. No assets will be selected. Returning empty portfolio.")
            return {}
        elif max_assets_cap < num_available_assets:
            selected_tickers_for_portfolio = asset_tickers[:max_assets_cap]
            logger.info(
                f"{strategy_name}: Applied 'max_assets_cap' of {max_assets_cap}. Selected {len(selected_tickers_for_portfolio)} assets from {num_available_assets} available.")

    num_selected_assets_final = len(selected_tickers_for_portfolio)

    if num_selected_assets_final == 0:

        logger.warning(
            f"{strategy_name}: No assets selected for the portfolio after applying filters/caps. Returning empty portfolio.")
        return {}

    equal_weight = 1.0 / num_selected_assets_final

    portfolio_weights_dict = {ticker: equal_weight for ticker in selected_tickers_for_portfolio}

    logger.info(
        f"{strategy_name}: Successfully constructed portfolio with {num_selected_assets_final} assets. Each asset assigned a weight of approximately {equal_weight:.6f}.")
    logger.debug(f"{strategy_name}: Final portfolio weights: {portfolio_weights_dict}")

    return portfolio_weights_dict