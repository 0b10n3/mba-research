# visualizations.py
"""
Módulo para Geração de Gráficos e Visualizações.

Este arquivo contém funções para plotar os resultados das simulações,
como a evolução do patrimônio dos portfólios e a comparação de métricas.
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from logger import log


def plot_portfolio_performance(portfolio_values: pd.DataFrame, output_dir: str, filename_base: str):
    """
    Plota e salva um gráfico da evolução do patrimônio dos portfólios e do benchmark.

    Args:
        portfolio_values (pd.DataFrame): DataFrame com o valor de cada portfólio ao longo do tempo.
        output_dir (str): Diretório para salvar o gráfico.
        filename_base (str): Nome base para o arquivo de imagem.
    """
    log.info("Gerando gráfico de desempenho dos portfólios...")

    # Normaliza os valores para começar em 1, facilitando a comparação.
    normalized_values = portfolio_values / portfolio_values.iloc[0]

    plt.style.use('seaborn-v0_8-darkgrid')
    fig, ax = plt.subplots(figsize=(14, 8))

    # Plota cada portfólio
    for column in normalized_values.columns:
        ax.plot(normalized_values.index, normalized_values[column], label=column)

    ax.set_title('Desempenho Comparativo dos Portfólios (Base 1)', fontsize=16)
    ax.set_xlabel('Data', fontsize=12)
    ax.set_ylabel('Valor Normalizado', fontsize=12)
    ax.legend(title='Estratégias', fontsize=10)
    ax.grid(True)

    plt.tight_layout()

    # Salva o gráfico em um arquivo.
    filepath = os.path.join(output_dir, f'{filename_base}_performance.png')
    try:
        plt.savefig(filepath, dpi=300)
        log.info(f"Gráfico de desempenho salvo em: {filepath}")
    except Exception as e:
        log.error(f"Erro ao salvar o gráfico: {e}")
    plt.close(fig)


def plot_risk_return_scatter(metrics_df: pd.DataFrame, output_dir: str, filename_base: str):
    """
    Plota e salva um gráfico de dispersão (scatter plot) de Risco vs. Retorno.

    Args:
        metrics_df (pd.DataFrame): DataFrame com as métricas calculadas para cada portfólio.
        output_dir (str): Diretório para salvar o gráfico.
        filename_base (str): Nome base para o arquivo de imagem.
    """
    log.info("Gerando gráfico de Risco vs. Retorno...")

    # Seleciona as métricas necessárias.
    risk = metrics_df.loc['Volatilidade Anualizada (%)']
    ret = metrics_df.loc['Retorno Anualizado (%)']

    plt.style.use('seaborn-v0_8-darkgrid')
    fig, ax = plt.subplots(figsize=(10, 7))

    ax.scatter(risk, ret, s=100, c='blue', alpha=0.7)

    # Adiciona rótulos para cada ponto.
    # *** CORREÇÃO APLICADA AQUI ***
    # Corrigido o FutureWarning, usando .iloc para acessar por posição inteira.
    for i, txt in enumerate(risk.index):
        ax.annotate(txt, (risk.iloc[i], ret.iloc[i]), xytext=(5, 5), textcoords='offset points')

    ax.set_title('Risco (Volatilidade) vs. Retorno Anualizado', fontsize=16)
    ax.set_xlabel('Volatilidade Anualizada (%)', fontsize=12)
    ax.set_ylabel('Retorno Anualizado (%)', fontsize=12)
    ax.grid(True)

    plt.tight_layout()

    # Salva o gráfico em um arquivo.
    filepath = os.path.join(output_dir, f'{filename_base}_risk_return.png')
    try:
        plt.savefig(filepath, dpi=300)
        log.info(f"Gráfico de Risco vs. Retorno salvo em: {filepath}")
    except Exception as e:
        log.error(f"Erro ao salvar o gráfico: {e}")
    plt.close(fig)

