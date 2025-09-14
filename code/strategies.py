# strategies.py
"""
Módulo de Estratégias de Alocação de Portfólio.

Cada função neste arquivo implementa uma estratégia de alocação de ativos,
conforme definido na metodologia da pesquisa. As funções recebem os preços
históricos e retornam um dicionário com os pesos ótimos para cada ativo.
"""
import numpy as np
import pandas as pd
from pypfopt import EfficientFrontier, risk_models, expected_returns
import riskparityportfolio as rpp

from logger import log


def equal_weights_strategy(assets: list, **kwargs) -> dict:
    """
    Estratégia de alocação com pesos iguais (1/N).

    Args:
        assets (list): Lista de tickers dos ativos.
        **kwargs: Argumentos adicionais para compatibilidade de interface.

    Returns:
        dict: Dicionário com os pesos alocados para cada ativo.
    """
    log.info("Calculando pesos para a estratégia Equal Weights (1/N).")
    num_assets = len(assets)
    weights = {asset: 1 / num_assets for asset in assets}
    return weights


def markowitz_strategy(prices: pd.DataFrame, **kwargs) -> dict:
    """
    Estratégia de otimização de Média-Variância (Markowitz) com shrinkage.
    Busca o portfólio com o máximo Índice de Sharpe.

    Args:
        prices (pd.DataFrame): DataFrame com preços históricos dos ativos.

    Returns:
        dict: Dicionário com os pesos ótimos para cada ativo.
    """
    log.info("Calculando pesos para a estratégia Markowitz (Max Sharpe).")
    try:
        # 1. Calcular retornos esperados (mu).
        mu = expected_returns.mean_historical_return(prices, compounding=True, frequency=252)

        # 2. Calcular a matriz de covariância com shrinkage de Ledoit & Wolf.
        S = risk_models.CovarianceShrinkage(prices).ledoit_wolf()

        # 3. Otimização para o máximo Índice de Sharpe.
        ef = EfficientFrontier(mu, S)
        ef.max_sharpe()

        # Obtém os pesos limpos (remove pesos muito pequenos).
        weights = dict(ef.clean_weights())
        return weights
    except Exception as e:
        log.error(f"Erro na estratégia Markowitz: {e}. Retornando pesos iguais como fallback.")
        return equal_weights_strategy(assets=prices.columns.tolist())


def risk_parity_strategy(prices: pd.DataFrame, **kwargs) -> dict:
    """
    Estratégia de Paridade de Risco (Risk Parity).
    Aloca o capital de forma que a contribuição de risco de cada ativo seja igual.

    Args:
        prices (pd.DataFrame): DataFrame com preços históricos dos ativos.

    Returns:
        dict: Dicionário com os pesos ótimos para cada ativo.
    """
    log.info("Calculando pesos para a estratégia Risk Parity.")
    try:
        # Calcula os retornos diários a partir dos preços.
        returns = prices.pct_change().dropna()

        # Calcula a matriz de covariância dos retornos.
        cov_matrix = returns.cov()

        # *** CORREÇÃO APLICADA AQUI ***
        # A função 'design' espera um array numpy, não um DataFrame pandas.
        # Convertendo o DataFrame para um array numpy com .values
        weights_array = rpp.vanilla.design(cov_matrix.values)

        weights = dict(zip(prices.columns, weights_array))
        return weights
    except Exception as e:
        log.error(f"Erro na estratégia Risk Parity: {e}. Retornando pesos iguais como fallback.")
        return equal_weights_strategy(assets=prices.columns.tolist())


def goal_based_investing_strategy(prices: pd.DataFrame, metas_gbi: dict, **kwargs) -> dict:
    """
    Estratégia de Goal-Based Investing (GBI).
    Combina sub-portfólios otimizados para diferentes metas.

    Args:
        prices (pd.DataFrame): DataFrame com preços históricos dos ativos.
        metas_gbi (dict): Dicionário de configuração das metas.

    Returns:
        dict: Dicionário com os pesos agregados para cada ativo.
    """
    log.info("Calculando pesos para a estratégia Goal-Based Investing (GBI).")

    final_weights = {asset: 0.0 for asset in prices.columns}

    try:
        for meta_nome, meta_info in metas_gbi.items():
            log.info(f"Otimizando para a meta: {meta_nome}")

            meta_assets = meta_info['ativos']
            meta_ponderacao = meta_info['ponderacao']

            # Filtra os preços apenas para os ativos desta meta.
            meta_prices = prices[meta_assets]

            # Otimiza o sub-portfólio da meta (aqui usamos Markowitz, mas poderia ser outra).
            # Esta é uma escolha de implementação, a otimização de cada sub-portfólio
            # pode ser diferente (ex: min_volatility para 'Segurança').
            meta_weights = markowitz_strategy(prices=meta_prices)

            # Adiciona os pesos do sub-portfólio aos pesos finais, ponderados pela meta.
            for asset, weight in meta_weights.items():
                final_weights[asset] += weight * meta_ponderacao

        return final_weights
    except Exception as e:
        log.error(f"Erro na estratégia GBI: {e}. Retornando pesos iguais como fallback.")
        return equal_weights_strategy(assets=prices.columns.tolist())


# Dicionário que mapeia nomes de estratégias às suas respectivas funções.
# Facilita a chamada dinâmica no script principal.
STRATEGY_FUNCTIONS = {
    'Equal Weights': equal_weights_strategy,
    'Markowitz': markowitz_strategy,
    'Risk Parity': risk_parity_strategy,
    'Goal-Based Investing': goal_based_investing_strategy,
}
