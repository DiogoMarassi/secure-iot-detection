# Amostra 5%

Amostragem simples de **5% de cada cenário** do dataset IoT-23, sem nenhum tratamento de balanceamento.

## Como foi gerado

O script `gerar_amostra.py` percorre os 23 cenários e retira aleatoriamente 5% das linhas de cada arquivo `conn.log.labeled`, concatenando tudo em um único CSV.

## Características

- **Arquivo:** `dataset_5pct_sem_tratamento.csv`
- **Semente aleatória:** 42 (reproduzível)
- **Colunas:** todas as 23 colunas originais + coluna `scenario` (identifica o cenário de origem)

## Limitação

O desbalanceamento original é preservado proporcionalmente. Cenários com dezenas de milhões de flows dominam o dataset; classes raras podem ter pouquíssimos exemplos.
