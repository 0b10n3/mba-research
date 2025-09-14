# config.py
"""
Arquivo de Configuração Central para a Simulação de Portfólios.

Este arquivo armazena todos os parâmetros e configurações utilizados no backtesting,
facilitando a alteração de premissas e a execução de diferentes cenários
sem a necessidade de modificar o código-fonte principal da lógica da simulação.
"""

# -- PARÂMETROS GERAIS DA SIMULAÇÃO --

# Período de análise para o backtesting.
# Formato: 'YYYY-MM-DD'
START_DATE = '2014-01-01'
END_DATE = '2024-12-31'

# Valor inicial do portfólio para a simulação.
CAPITAL_INICIAL = 100000.00

# Frequência de rebalanceamento dos portfólios em meses.
# Ex: 3 = rebalanceamento trimestral.
REBALANCEAMENTO_MESES = 3

# Ativo de referência (benchmark) para comparação de desempenho.
# O ticker deve ser compatível com o Yahoo Finance.
BENCHMARK_TICKER = '^BVSP'

# -- LISTA DE ATIVOS (ETFs) --
# Universo de ETFs selecionados para a construção dos portfólios.
# Os tickers devem ser compatíveis com o Yahoo Finance (adicionar '.SA' para ativos brasileiros).
# Esta lista é crucial e deve refletir o universo de ativos definido na metodologia da pesquisa.
ASSETS = [
    'BOVA11.SA',  # iShares Ibovespa Fundo de Índice
    'SMAL11.SA',  # iShares Small Cap Fundo de Índice
    'IVVB11.SA',  # iShares S&P 500 Fundo de Índice
    'IMAB11.SA',  # It Now IMA-B Fundo de Índice
    'GOLD11.SA',  # BTG Pactual Ouro Fundo de Índice
    'PIBB11.SA',  # It Now IBrX-50 Fundo de Índice
]

# -- PARÂMETROS PARA ESTRATÉGIAS ESPECÍFICAS --

# Taxa de juros livre de risco (média anualizada) para o período.
# Usada no cálculo dos Índices de Sharpe, Sortino e Alfa de Jensen.
# Este valor deve ser apurado conforme a metodologia (ex: média da Selic ou CDI no período).
TAXA_LIVRE_RISCO_ANUAL = 0.10 # Valor exemplo, substituir pelo valor apurado na pesquisa.

# Alvo de retorno para o Índice de Sortino.
# Geralmente, utiliza-se a taxa livre de risco.
TARGET_RETURN_SORTINO_ANUAL = TAXA_LIVRE_RISCO_ANUAL

# -- CONFIGURAÇÕES PARA O GOAL-BASED INVESTING (GBI) --

# Definição das metas (sub-portfólios).
# Cada chave é o nome da meta. O valor é um dicionário com:
# 'ponderacao': O percentual do capital total alocado para esta meta.
# 'ativos': A lista de ETFs que compõem o universo desta meta específica.
METAS_GBI = {
    'SEGURANCA': {
        'ponderacao': 0.40,  # 40% do capital total
        'ativos': ['IMAB11.SA', 'GOLD11.SA'] # Foco em ativos de menor volatilidade
    },
    'CRESCIMENTO': {
        'ponderacao': 0.60,  # 60% do capital total
        'ativos': ['BOVA11.SA', 'SMAL11.SA', 'IVVB11.SA', 'PIBB11.SA'] # Foco em ativos de maior potencial de retorno
    }
}
# A soma das ponderações deve ser 1.0 (ou 100%).
assert sum(meta['ponderacao'] for meta in METAS_GBI.values()) == 1.0, "A soma das ponderações das metas GBI deve ser 1."


# -- CONFIGURAÇÕES DE SAÍDA DE DADOS --

# Diretório para salvar os resultados (CSV e gráficos).
OUTPUT_DIR = 'results'

# Nome base para os arquivos de saída.
OUTPUT_FILENAME_BASE = f'resultados_simulacao_{START_DATE}_a_{END_DATE}'

