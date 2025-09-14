# main.py
"""
Script Principal para Execução do Backtesting de Portfólios.

Este script orquestra todo o processo de simulação:
1.  Carrega as configurações do arquivo `config.py`.
2.  Busca os dados de mercado necessários usando `fn.py`.
3.  Executa o loop de backtesting, rebalanceando os portfólios periodicamente.
4.  Para cada período de rebalanceamento, calcula os pesos ótimos usando as
    funções de `strategies.py`.
5.  Calcula os retornos e o valor acumulado de cada portfólio.
6.  Ao final da simulação, calcula as métricas de desempenho consolidadas.
7.  Salva os resultados em arquivos CSV e gera visualizações gráficas
    usando `visualizations.py`.
"""
import pandas as pd
import numpy as np
import os

# Importa as configurações e módulos do projeto
import config
from logger import log
from fn import get_market_data, calculate_performance_metrics
from strategies import STRATEGY_FUNCTIONS
from visualizations import plot_portfolio_performance, plot_risk_return_scatter


def run_backtest():
    """
    Executa o processo completo de backtesting.
    """
    log.info("=" * 50)
    log.info("INICIANDO SIMULAÇÃO DE ESTRATÉGIAS DE PORTFÓLIO")
    log.info("=" * 50)

    # --- 1. CARREGAMENTO DE DADOS ---
    all_tickers = config.ASSETS + [config.BENCHMARK_TICKER]
    market_data = get_market_data(all_tickers, config.START_DATE, config.END_DATE)

    if market_data.empty:
        log.error("Não foi possível obter os dados de mercado. Abortando a simulação.")
        return

    asset_prices = market_data[config.ASSETS]
    benchmark_prices = market_data[config.BENCHMARK_TICKER]

    asset_returns = asset_prices.pct_change().dropna()
    benchmark_returns = benchmark_prices.pct_change().dropna()

    # --- 2. INICIALIZAÇÃO DOS PORTFÓLIOS ---
    log.info("Inicializando portfólios...")
    portfolio_values = pd.DataFrame(index=market_data.index, columns=STRATEGY_FUNCTIONS.keys(), dtype=float)
    portfolio_values.iloc[0] = config.CAPITAL_INICIAL

    weights_history = {name: [] for name in STRATEGY_FUNCTIONS.keys()}
    current_weights = {}

    # Define as datas de rebalanceamento
    rebalance_dates = asset_returns.resample(f'{config.REBALANCEAMENTO_MESES}ME').first().index

    # --- 3. EXECUÇÃO DO LOOP DE BACKTESTING ---
    for i, date in enumerate(asset_returns.index):
        # Data do dia anterior para obter o valor do portfólio no início do dia
        previous_date = market_data.index[market_data.index.get_loc(date) - 1]

        # --- Rebalanceamento ---
        # Rebalanceia no primeiro dia ou nas datas de rebalanceamento agendadas.
        if i == 0 or date in rebalance_dates:
            log.info(f"--- Rebalanceamento em {date.date()} ---")
            # Usa o histórico de preços até o dia anterior para a otimização.
            prices_for_opt = asset_prices.loc[:previous_date]

            for name, func in STRATEGY_FUNCTIONS.items():
                current_weights[name] = func(
                    prices=prices_for_opt,
                    assets=config.ASSETS,
                    metas_gbi=config.METAS_GBI
                )
                weights_history[name].append({'date': date, 'weights': current_weights[name]})

        # --- Atualização de Valor do Portfólio ---
        for name in STRATEGY_FUNCTIONS.keys():
            prev_value = portfolio_values.loc[previous_date, name]

            todays_returns = asset_returns.loc[date]
            weights_vector = pd.Series(current_weights[name])

            # Alinha os pesos com os retornos para garantir consistência
            aligned_returns, aligned_weights = todays_returns.align(weights_vector, join='right', fill_value=0)
            portfolio_return = np.dot(aligned_returns, aligned_weights)

            portfolio_values.loc[date, name] = prev_value * (1 + portfolio_return)

    # Preenche quaisquer dias não negociados com o último valor válido
    portfolio_values.ffill(inplace=True)

    # --- 4. CÁLCULO DAS MÉTRICAS FINAIS ---
    log.info("Simulação concluída. Calculando métricas finais de desempenho...")

    # Adiciona o benchmark para comparação, garantindo que o dia inicial esteja presente
    benchmark_values = (1 + benchmark_returns).cumprod() * config.CAPITAL_INICIAL
    initial_day_value = pd.Series({market_data.index[0]: config.CAPITAL_INICIAL})
    benchmark_values = pd.concat([initial_day_value, benchmark_values])

    portfolio_values['Benchmark (IBOVESPA)'] = benchmark_values.ffill()

    portfolio_daily_returns = portfolio_values.pct_change().dropna()

    final_metrics = {}
    for name in portfolio_values.columns:  # Itera sobre todas as colunas, incluindo o benchmark
        final_metrics[name] = calculate_performance_metrics(
            portfolio_daily_returns[name],
            config.TAXA_LIVRE_RISCO_ANUAL,
            config.TARGET_RETURN_SORTINO_ANUAL,
            portfolio_daily_returns['Benchmark (IBOVESPA)']
        )

    metrics_df = pd.DataFrame(final_metrics)

    # --- 5. SALVANDO OS RESULTADOS ---
    log.info("Salvando resultados em arquivos CSV e gerando gráficos...")
    if not os.path.exists(config.OUTPUT_DIR):
        os.makedirs(config.OUTPUT_DIR)

    metrics_filepath = os.path.join(config.OUTPUT_DIR, f'{config.OUTPUT_FILENAME_BASE}_metrics.csv')
    metrics_df.to_csv(metrics_filepath, decimal=',', sep=';')
    log.info(f"Métricas salvas em: {metrics_filepath}")

    values_filepath = os.path.join(config.OUTPUT_DIR, f'{config.OUTPUT_FILENAME_BASE}_values.csv')
    portfolio_values.to_csv(values_filepath, decimal=',', sep=';')
    log.info(f"Evolução do patrimônio salva em: {values_filepath}")

    plot_portfolio_performance(portfolio_values, config.OUTPUT_DIR, config.OUTPUT_FILENAME_BASE)
    plot_risk_return_scatter(metrics_df, config.OUTPUT_DIR, config.OUTPUT_FILENAME_BASE)

    log.info("=" * 50)
    log.info("PROCESSO CONCLUÍDO COM SUCESSO!")
    log.info("=" * 50)

    print("\n--- RESUMO DAS MÉTRICAS ---")
    print(metrics_df.round(4))
    print("\n")


if __name__ == '__main__':
    run_backtest()
