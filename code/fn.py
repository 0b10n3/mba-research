# fn.py
"""
Módulo de Funções Auxiliares.

Este arquivo centraliza as funções para:
1.  Coleta de dados de mercado (preços de ativos).
2.  Cálculo de métricas de desempenho e risco de portfólios.

As funções aqui presentes são utilizadas pelo script principal (main.py) e
pelas estratégias de alocação (strategies.py).
"""
import pandas as pd
import numpy as np
import yfinance as yf
from typing import List, Dict

# Importa a instância do logger configurado.
from logger import log

def get_market_data(tickers: List[str], start_date: str, end_date: str) -> pd.DataFrame:
    """
    Busca dados históricos de preços de fechamento ajustados do Yahoo Finance.

    Args:
        tickers (List[str]): Lista de tickers dos ativos.
        start_date (str): Data de início no formato 'YYYY-MM-DD'.
        end_date (str): Data de fim no formato 'YYYY-MM-DD'.

    Returns:
        pd.DataFrame: DataFrame com os preços de fechamento ajustados,
                      onde cada coluna corresponde a um ticker.
                      Retorna um DataFrame vazio em caso de erro.
    """
    log.info(f"Buscando dados de mercado para {len(tickers)} ativos de {start_date} a {end_date}.")
    try:
        # Baixa os dados usando a biblioteca yfinance.
        # O parâmetro auto_adjust=True (padrão recente do yfinance) já retorna
        # os preços de fechamento ajustados na coluna 'Close'.
        data = yf.download(tickers, start=start_date, end=end_date, progress=False, auto_adjust=True)

        # Seleciona apenas os preços de fechamento (que já são ajustados).
        # Se baixar apenas um ticker, o yfinance não retorna um MultiIndex nas colunas.
        if len(tickers) == 1:
            prices = data[['Close']]
            # Renomeia a coluna para o nome do ticker para manter a consistência
            prices.columns = tickers
        else:
            prices = data['Close']

        if prices.empty:
            log.warning("O DataFrame de preços retornado pelo yfinance está vazio.")
            return pd.DataFrame()

        # Verifica se todos os tickers solicitados estão presentes nas colunas do DataFrame resultante.
        missing_tickers = set(tickers) - set(prices.columns)
        if missing_tickers:
            log.warning(f"Não foi possível obter dados para todos os tickers. Ausentes: {', '.join(missing_tickers)}")
            # Para garantir a consistência do universo de ativos, retorna um DataFrame vazio se algum ativo faltar.
            return pd.DataFrame()

        # Remove linhas que contenham qualquer valor NaN (dias não negociados para algum ativo).
        prices.dropna(inplace=True)
        log.info(f"Dados obtidos com sucesso. {len(prices)} pregões válidos encontrados.")
        return prices

    except Exception as e:
        log.error(f"Ocorreu um erro ao buscar os dados de mercado: {e}")
        return pd.DataFrame()


def calculate_performance_metrics(portfolio_returns: pd.Series, risk_free_rate_annual: float, target_return_annual: float, benchmark_returns: pd.Series) -> Dict[str, float]:
    """
    Calcula um conjunto de métricas de desempenho e risco para um portfólio.

    Args:
        portfolio_returns (pd.Series): Série de retornos diários do portfólio.
        risk_free_rate_annual (float): Taxa de juros livre de risco anualizada.
        target_return_annual (float): Taxa de retorno alvo anualizada para o Índice de Sortino.
        benchmark_returns (pd.Series): Série de retornos diários do benchmark (ex: IBOVESPA).

    Returns:
        Dict[str, float]: Dicionário contendo as métricas calculadas.
    """
    # Fator de anualização baseado no número de dias de negociação em um ano.
    ANNUALIZATION_FACTOR = 252

    # -- Retorno Total e Anualizado --
    total_return = (1 + portfolio_returns).prod() - 1
    annualized_return = ((1 + total_return) ** (ANNUALIZATION_FACTOR / len(portfolio_returns))) - 1

    # -- Volatilidade Anualizada --
    annualized_volatility = portfolio_returns.std() * np.sqrt(ANNUALIZATION_FACTOR)

    # -- Índice de Sharpe --
    # Converte a taxa livre de risco anual para diária.
    risk_free_rate_daily = (1 + risk_free_rate_annual)**(1/ANNUALIZATION_FACTOR) - 1
    excess_returns = portfolio_returns - risk_free_rate_daily
    sharpe_ratio = (excess_returns.mean() / excess_returns.std()) * np.sqrt(ANNUALIZATION_FACTOR) if excess_returns.std() != 0 else 0.0

    # -- Índice de Sortino --
    target_return_daily = (1 + target_return_annual)**(1/ANNUALIZATION_FACTOR) - 1
    downside_returns = portfolio_returns[portfolio_returns < target_return_daily]
    downside_std = downside_returns.std() * np.sqrt(ANNUALIZATION_FACTOR)
    sortino_ratio = (annualized_return - target_return_annual) / downside_std if downside_std != 0 else 0.0

    # -- Maximum Drawdown (MDD) --
    cumulative_returns = (1 + portfolio_returns).cumprod()
    peak = cumulative_returns.expanding(min_periods=1).max()
    drawdown = (cumulative_returns - peak) / peak
    max_drawdown = drawdown.min()

    # -- Alfa de Jensen --
    # Alinha os retornos do portfólio e do benchmark.
    merged_returns = pd.DataFrame({'portfolio': portfolio_returns, 'benchmark': benchmark_returns}).dropna()
    # Calcula o Beta do portfólio.
    covariance = merged_returns['portfolio'].cov(merged_returns['benchmark'])
    benchmark_variance = merged_returns['benchmark'].var()
    beta = covariance / benchmark_variance if benchmark_variance != 0 else 0.0

    # Calcula o retorno esperado pelo CAPM e o Alfa.
    expected_return_capm = risk_free_rate_annual + beta * (benchmark_returns.mean() * ANNUALIZATION_FACTOR - risk_free_rate_annual)
    jensen_alpha = annualized_return - expected_return_capm


    metrics = {
        'Retorno Total Acumulado (%)': total_return * 100,
        'Retorno Anualizado (%)': annualized_return * 100,
        'Volatilidade Anualizada (%)': annualized_volatility * 100,
        'Índice de Sharpe': sharpe_ratio,
        'Índice de Sortino': sortino_ratio,
        'Maximum Drawdown (MDD) (%)': max_drawdown * 100,
        'Alfa de Jensen (%)': jensen_alpha * 100,
        'Beta': beta
    }

    return metrics
