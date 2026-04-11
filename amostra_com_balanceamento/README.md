# Amostra com Balanceamento

Amostragem inteligente do dataset IoT-23 que garante a presença de classes raras e equaliza as classes abundantes.

## Como foi gerado

O script `gerar_amostra_balanceada.py` classifica cada cenário em Tipo A ou Tipo B com base no volume total de flows, aplicando uma estratégia diferente para cada um.

### Tipo A — Cenários pequenos (< 1.000.000 flows)
Todos os flows são incluídos (100%). São cenários com poucos dados e descartá-los seria perda de informação.

### Tipo B — Cenários grandes (>= 1.000.000 flows)
Aplica lógica de balanceamento por classe dentro do limite de 5% do total do cenário:

1. **Classes raras** (< 250.000 flows): incluídas 100%, garantindo sua presença no dataset.
2. **Classes não raras** (>= 250.000 flows): equalizadas para o mesmo número de linhas (baseado na menor delas), e amostradas igualmente até completar o orçamento de 5%.

## Características

- **Arquivo:** `dataset_balanceado.csv`
- **Semente aleatória:** 42 (reproduzível)
- **Colunas:** todas as 23 colunas originais + coluna `scenario` (identifica o cenário de origem)
- **Logs:** `logs_balanceamento.txt` contém o detalhamento de quantas linhas foram coletadas por classe em cada cenário.

## Vantagem sobre a amostra simples

Classes raras nunca são perdidas por azar da amostragem. Classes dominantes não distorcem o dataset. Mais adequado para treinar modelos de classificação multiclasse.
