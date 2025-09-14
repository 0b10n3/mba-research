Aluno(a): Silvano Antonio Alves Pereira Junior
Orientador(a): Regiane Batista dos Santos
Curso: MBA em Finanças e Controladoria

Eficácia do Goal-Based Investing com ETFs versus estratégias tradicionais de alocação de portfólio


Introdução

O mercado financeiro contemporâneo apresenta uma variedade de abordagens para alocação de ativos, com estratégias tradicionais como equal weights, otimização de Markowitz e paridade de risco (risk parity) constituindo os pilares fundamentais para investidores institucionais e individuais. Estas metodologias, embora amplamente adotadas, frequentemente priorizam métricas abstratas de otimização em detrimento das necessidades específicas dos investidores (Chhabra, 2005). Neste contexto, a metodologia de Goal-Based Investing (GBI) emerge como uma alternativa promissora que reorienta o processo de investimento, centralizando as decisões de alocação nos objetivos financeiros concretos de cada investidor.
Markowitz (1952) apresenta as bases da teoria moderna de portfólio, introduzindo o conceito de diversificação. Ele demonstra que os investidores podem reduzir o risco combinando ativos cujos retornos não estão perfeitamente correlacionados. Além disso, desenvolve a ideia da fronteira eficiente, que representa o conjunto de portfólios que oferecem o maior retorno esperado para um dado nível de risco.
O Goal-Based Investing, conforme estabelecido por Brunel (2015), fundamenta-se na premissa de que as decisões de investimento devem alinhar-se diretamente a metas financeiras, com horizontes temporais específicos e níveis individualizados de tolerância a risco. Este paradigma contrasta significativamente com abordagens tradicionais que tendem a tratar os investidores como entidades homogêneas, aplicando soluções padronizadas que podem não corresponder às suas particularidades. 
Mais recentemente, Markowitz expandiu sua teoria para incorporar insights das finanças comportamentais. Em Das et al. (2010), ele reconhece que os investidores, influenciados por vieses cognitivos, nem sempre agem de forma completamente racional. Essa perspectiva dialoga com a teoria da perspectiva de Kahneman e Tversky (1979), que descreve a avaliação assimétrica de ganhos e perdas, evidenciando como fatores psicológicos afetam as decisões financeiras. Essa integração reforça a relevância de considerar o comportamento humano real na gestão de portfólios, um princípio central no GBI.
Este paradigma contrasta significativamente com abordagens tradicionais que tendem a tratar os investidores como entidades homogêneas, aplicando soluções padronizadas que podem não corresponder às suas particularidades. Conforme apontado por Nevins (2004), a integração entre finanças tradicionais e comportamentais proposta pelo GBI representa uma evolução natural na gestão de portfólios, ampliando o escopo de análise para além dos aspectos puramente quantitativos.
Paralelamente à ascensão do GBI, os Exchange-Traded Funds (ETFs) têm se destacado como veículos de investimento versáteis. Das e Chandani (2018) apontam que os ETFs oferecem diversificação, liquidez, transparência e custos reduzidos, características que os tornam adequados para implementar estratégias de GBI. Ao possibilitar a construção de portfólios personalizados com base em metas específicas, os ETFs facilitam a execução eficiente dessa abordagem.
A estratégia de equal weights, também conhecida como 1/N, tem sido objeto de crescente interesse acadêmico e prático. Em um estudo seminal, DeMiguel et al. (2009) investigaram sua eficiência em comparação com métodos de otimização mais complexos, como o de Markowitz. Surpreendentemente, demonstraram que, em diversos cenários, a alocação igualitária não apenas compete, mas por vezes supera estratégias teoricamente mais sofisticadas em desempenho ajustado ao risco. Esse resultado desafia a crença convencional de que maior complexidade implica melhores resultados, evidenciando a robustez e a simplicidade da equal weights como vantagens práticas.
A risk parity destaca-se por adaptar a alocação de ativos a diferentes condições de mercado, oferecendo uma diversificação mais robusta que métodos tradicionais. Ganhou relevância após a crise financeira de 2008, quando investidores buscaram reduzir riscos sistêmicos sem comprometer retornos. Asness et al. (2012) realizam uma análise, mostrando que a risk parity, quando otimizada com alavancagem, pode superar portfólios tradicionais em eficiência de risco-retorno.
Esta pesquisa busca comparar empiricamente o desempenho histórico de portfólios construídos sob os princípios do Goal-Based Investing e compostos predominantemente por ETFs, com portfólios estruturados segundo metodologias tradicionais de alocação. 
Ao investigar comparativamente estas diferentes abordagens, este estudo visa contribuir não apenas para o avanço do conhecimento acadêmico, mas também para o aperfeiçoamento das práticas de gestão de portfólio no mercado brasileiro.
Para alcançar este objetivo, será realizada uma análise empírica utilizando dados históricos de desempenho de portfólios. Os dados serão coletados de bases de dados financeiras confiáveis, como B3, ANBIMA e CVM, cobrindo o período de 2014 a 2024. Esses dados serão analisados por meio de técnicas estatísticas de comparação de desempenho, como comparação dos retornos totais e anualizados, comparação de importantes indicadores de desempenho como: volatilidade, máximo drawdown, indíces de Sharpe e Sortino, Alfa de Jensen.

Objetivo
O presente estudo tem como objetivo comparar a eficácia do Goal-Based Investing com ETFs e das estratégias tradicionais de alocação de portfólio.

Material e Métodos
	Esta pesquisa caracteriza-se como quantitativa, de natureza aplicada e com abordagem comparativa, utilizando procedimentos experimentais através de simulação computacional. Conforme Gil (2017). O delineamento temporal será longitudinal retrospectivo, abrangendo o período de 2014 a 2024, permitindo a avaliação das estratégias de investimento em diferentes ciclos econômicos.
No levantamento de dados serão coletados dados de fontes confiáveis como ANBIMA, B3, CVM e IPEA,  incluindo:
    • Séries históricas de preços e retornos de ETFs negociados no mercado brasileiro com periodicidade diária.
    • Séries históricas de índices de referência (benchmarks) setoriais e amplos de mercado
    • Indicadores econômicos e financeiros relevantes (taxa de juros, inflação, PIB, produção industrial).
    • Métricas de desempenho e risco de fundos de investimento tradicionais para contextualização.
    • 
O processamento e análise dos dados serão realizados utilizando a linguagem de programação Python, com bibliotecas de análise de dados como pandas, numpy e scipy. A visualização dos resultados será implementada através das bibliotecas matplotlib e seaborn. 

	Na estapa de construção dos portfólios serão elaborados quatro tipos de portfólios para comparação:

    • Portfólio Equal Weights: Alocação com pesos iguais entre classes de ativos selecionadas. Rebalanceamento trimestral para manutenção dos pesos iguais. Implementação conforme metodologia proposta por DeMiguel et al. (2009)
    • Portfólio Markowitz: Alocação baseada na otimização média-variância conforme o modelo clássico. Rebalanceamento trimestral. Utilização de técnicas de regularização da matriz de covariância (Ledoit & Wolf, 2004).
    • Portfólio Risk Parity:  Alocação fundamentada na distribuição equitativa do risco entre diferentes ativos. Rebalanceamento trimestral. Implementação segundo o método proposto por Maillard et al. (2010). 
    • Portfólio Goal-Based: Construído segundo os princípios do Goal-Based Investing (Das et al., 2010). Rebalanceamento trimestral. Segmentação em sub-portfólios dedicados a objetivos específicos. Seleção de ETFs alinhados às características de cada objetivo

Para cada metodologia, serão utilizados predominantemente ETFs como veículos de investimento, garantindo assim a comparabilidade entre as diferentes estratégias. O universo de ETFs será  selecionado para incluir instrumentos que representem adequadamente as principais classes de ativos, setores econômicos e regiões geográficas, com preferência para ETFs com histórico mais longo, maior liquidez e menores custos operacionais.

Quanto a análise de desempenho dos portfólios, serão avaliados os seguintes indicadores:
    • Retorno total e anualizado
Visa medir o ganho ou perda absoluto de um investimento em um período (Feibel, 2003). O retorno anualizado converte retornos de períodos irregulares para uma base anual, facilitando comparações.
Fórmula:
    • Retorno Total:
	$ R_t = \frac{V_f - V_i}{V_i} \times 100% $
    • Retorno Anualizado:
$ R_{anual} = \left( \prod_{t=1}^{n} (1 + R_t) \right)^{\frac{252}{n}} - 1 $

    • Volatilidade (desvio-padrão dos retornos)
Quantifica a variação dos retornos em relação à média, sendo uma forma comum de se medir o risco associado ao investimento (Neto, 2018). Valores mais altos sugerem maior incerteza.
Fórmula:
$ \sigma = \sqrt{\frac{1}{n-1} \sum_{t=1}^{n} (R_t - \overline{R})^2} $

    • Índices de Sharpe  
Avalia o retorno ajustado pelo risco total, comparando o excesso de retorno sobre a taxa livre de risco com a volatilidade do portfólio (Sharpe, 1966).
Fórmula:
$ Sharpe = \frac{R_p - R_f}{\sigma_p} $

    • Índices de Sortino.
Similar ao Sharpe, considera apenas o risco de queda (downside risk), Sortino e Price (1994), sendo mais adequado para avaliar estratégias com assimetria de retornos.
Fórmula:
$ Sortino = \frac{R_p - R_{alvo}}{\sigma_{down}} $

    • Maximum Drawdown.
Mede a maior perda percentual entre um pico e um vale subsequente, que reflete o risco extremo de um investimento.
Fórmula:
$ MDD = \frac{P_{pico} - P_{vale}}{P_{pico}} \times 100% $

    • Alfa de Jensen
Mede o retorno excedente em relação ao esperado pelo CAPM, indicando a habilidade do gestor em gerar performance (Neto, 2018).
Fórmula:
$ \alpha = R_p - [R_f + \beta (R_m - R_f)] $

    • Probabilidade de atingimento dos objetivos financeiros predeterminados (Das et al., 2010).

Resultados Esperados

Espera-se como resultado da pesquisa:

    1. Verificar qual foi returno total e anualizado de cada portfólio;
    2. Verificar o nível de volatilidade em cada portfólio ao longo do período análisado;
    3. Comparar os índices de Sharpe e Sortino dos portifólios;
    4. Verificar, por meio do Alfa de Jensen,  o retorno excedente em relação ao esperado pelo CAPM gerado pela aplicação de cada metodologia.


Cronograma de Atividades

Atividades planejadas	Mês
	1	2	3	4	5	6	7	8	9	10
Levantamento de bibliografia	X									
Definição do objetivo de pesquisa		X								
Elaboração do projeto de pesquisa			X							
Levantamento dos dados				X						
Análise dos dados					X					
Escrita dos resultados preliminares						X				
‍										
‍										
‍										
Escrita da versão final do TCC								X		
Revisão do TCC final									X	
Entrega do TCC final										X
‍										
Elaboração da apresentação										X
Entrega da apresentação										X


Referências 

Asness, C. S.; Frazzini, A.; Pedersen, L. H. 2012. Leverage Aversion and Risk Parity. Financial Analysts Journal 68(1): 47-59.
Brunel, J. L. P. 2015. Goals-Based Wealth Management: An Integrated and Practical Approach to Changing the Structure of Wealth Advisory Practices. Wiley Finance, Hoboken, NJ, EUA.
Chhabra, A. B. 2005. Beyond Markowitz: A Comprehensive Wealth Allocation Framework for Individual Investors. The Journal of Wealth Management 7(4): 8-34.
Das, S.; Chandani, A. 2018. Performance of Exchange Traded Funds (ETFs) in India. International Journal of Financial Management 8(1): 1-8.
Das, S.; Markowitz, H.; Scheid, J.; Statman, M. 2010. Portfolio Optimization with Mental Accounts. Journal of Financial and Quantitative Analysis 45(2): 311-334.
DeMiguel, V.; Garlappi, L.; Uppal, R. 2009. Optimal Versus Naive Diversification: How Inefficient is the 1/N Portfolio Strategy? The Review of Financial Studies 22(5): 1915-1953.
Feibel, B. J. 2003. Investment Performance Measurement. Wiley, Hoboken, NJ, EUA.

Gil, A. C. 2017. Como Elaborar Projetos de Pesquisa. 6ed. Atlas, São Paulo, SP, Brasil.
Kahneman, D.; Tversky, A. 1979. Prospect Theory: An Analysis of Decision under Risk. Econometrica 47(2): 263-291.
Ledoit, O.; Wolf, M. 2004. Honey, I Shrunk the Sample Covariance Matrix. The Journal of Portfolio Management 30(4): 110-119.
Maillard, S.; Roncalli, T.; Teïletché, J. 2010. The Properties of Equal Risk Contribution Portfolios. The Journal of Portfolio Management 36(4): 60-70.
Markowitz, H. 1952. Portfolio Selection. The Journal of Finance 7(1): 77-91.
Neto, A. A. 2018. Mercado Financeiro. 14ed. Atlas, São Paulo, SP, Brasil. 
Nevins, D. 2004. Goals-Based Investing: Integrating Traditional and Behavioral Finance. The Journal of Wealth Management 6(4): 8-23.
Sharpe, W. F. 1966. Mutual Fund Performance. The Journal of Business 39(1): 119-138. 
Sortino, F. A.; Price, L. N. 1994. Performance Measurement in a Downside Risk Framework. The Journal of Investing 3(3): 59-64.
