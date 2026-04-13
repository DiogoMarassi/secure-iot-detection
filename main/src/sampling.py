"""Amostragem estratificada por leitura em chunks para CSVs grandes."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd


def amostra_estratificada_por_chunks(
    caminho_csv: str,
    coluna_rotulo: str = "label",
    max_por_classe: int = 50_000,
    tamanho_chunk: int = 200_000,
    random_state: int = 42,
    encoding: str | None = None,
) -> pd.DataFrame:
    """Constrói uma amostra com até `max_por_classe` linhas por valor de rótulo.

    Percorre o arquivo em chunks para manter uso de memória controlado. Em cada
    chunk, para cada classe ainda abaixo do limite, sorteia linhas até completar
    a cota ou esgotar o chunk.

    Args:
        caminho_csv: Caminho do arquivo CSV.
        coluna_rotulo: Nome da coluna usada para estratificação (ex.: ``label``).
        max_por_classe: Número máximo de linhas por classe na amostra final.
        tamanho_chunk: Número de linhas por leitura com ``read_csv(chunksize=...)``.
        random_state: Semente para reprodutibilidade da amostragem e do embaralhamento final.
        encoding: Encoding do arquivo; ``None`` usa o padrão do pandas.

    Returns:
        DataFrame com a amostra estratificada, linhas em ordem aleatória.

    Raises:
        ValueError: Se a coluna de rótulo não existir em algum chunk.
        FileNotFoundError: Se o arquivo não existir (propagado pelo pandas).
    """
    rng = np.random.default_rng(random_state)
    coletados: dict[Any, list[pd.DataFrame]] = defaultdict(list)
    contagem: dict[Any, int] = defaultdict(int)

    kwargs: dict[str, Any] = {
        "chunksize": tamanho_chunk,
        "low_memory": False,
    }
    if encoding is not None:
        kwargs["encoding"] = encoding

    for chunk in pd.read_csv(caminho_csv, **kwargs):
        if coluna_rotulo not in chunk.columns:
            raise ValueError(
                f"Coluna '{coluna_rotulo}' não encontrada. Colunas: {list(chunk.columns)}"
            )
        for rotulo, grupo in chunk.groupby(coluna_rotulo, dropna=False):
            chave = rotulo if pd.notna(rotulo) else "__NA__"
            necessario = max_por_classe - contagem[chave]
            if necessario <= 0:
                continue
            fatia = grupo
            if len(fatia) > necessario:
                fatia = fatia.sample(
                    n=necessario, random_state=int(rng.integers(1_000_000_000))
                )
            coletados[chave].append(fatia)
            contagem[chave] += len(fatia)

    if not coletados:
        return pd.DataFrame()

    partes = [pd.concat(listas, ignore_index=True) for listas in coletados.values()]
    resultado = pd.concat(partes, ignore_index=True)
    return resultado.sample(frac=1.0, random_state=random_state).reset_index(drop=True)


def contagem_rotulos_por_chunks(
    caminho_csv: str,
    coluna_rotulo: str = "label",
    tamanho_chunk: int = 500_000,
    encoding: str | None = None,
) -> pd.Series:
    """Agrega contagens globais de cada rótulo sem carregar o arquivo inteiro.

    Args:
        caminho_csv: Caminho do CSV.
        coluna_rotulo: Coluna de classe.
        tamanho_chunk: Tamanho do chunk.
        encoding: Encoding opcional.

    Returns:
        Series indexada pelo rótulo com contagens.
    """
    kwargs: dict[str, Any] = {"chunksize": tamanho_chunk, "low_memory": False}
    if encoding is not None:
        kwargs["encoding"] = encoding

    total: dict[Any, int] = defaultdict(int)
    for chunk in pd.read_csv(caminho_csv, **kwargs):
        if coluna_rotulo not in chunk.columns:
            raise ValueError(f"Coluna '{coluna_rotulo}' não encontrada.")
        vc = chunk[coluna_rotulo].value_counts(dropna=False)
        for k, v in vc.items():
            total[k] += int(v)
    return pd.Series(total, dtype="int64").sort_values(ascending=False)
