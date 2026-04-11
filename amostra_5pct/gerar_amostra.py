"""
Gera uma amostra de 5% de cada cenário do dataset IoT-23 e salva em CSV.

Uso:
    python gerar_amostra.py

Saída:
    dataset_5pct.csv  (~2GB, ~16 milhões de linhas)
"""

import pandas as pd
import glob
import os

# ── Configurações ──────────────────────────────────────────────────────────────
BASE = "iot_23_datasets_small/opt/Malware-Project/BigDataset/IoTScenarios"
OUTPUT = "dataset_5pct.csv"
SAMPLE_RATE = 0.05
CHUNK_SIZE = 500_000   # linhas por chunk (controla uso de RAM)
RANDOM_SEED = 42

# Colunas após separação correta (as 3 últimas vêm juntas no TSV)
COLS_TSV = [
    "ts", "uid", "id.orig_h", "id.orig_p", "id.resp_h", "id.resp_p",
    "proto", "service", "duration", "orig_bytes", "resp_bytes",
    "conn_state", "local_orig", "local_resp", "missed_bytes", "history",
    "orig_pkts", "orig_ip_bytes", "resp_pkts", "resp_ip_bytes",
    "_last"          # contém: tunnel_parents   label   detailed-label
]
# ──────────────────────────────────────────────────────────────────────────────


def processar_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    """Separa a última coluna combinada e faz a amostragem."""
    # Divide "_last" em 3 colunas usando 3 espaços como separador
    split = chunk["_last"].str.split(r"\s{3}", expand=True)
    chunk["tunnel_parents"] = split[0]
    chunk["label"]          = split[1]
    chunk["detailed-label"] = split[2]
    chunk = chunk.drop(columns=["_last"])

    # Remove espaços extras que possam ter sobrado
    chunk["label"]          = chunk["label"].str.strip()
    chunk["detailed-label"] = chunk["detailed-label"].str.strip()

    return chunk.sample(frac=SAMPLE_RATE, random_state=RANDOM_SEED)


def main():
    arquivos = sorted(glob.glob(f"{BASE}/*/bro/conn.log.labeled"))
    print(f"Encontrados {len(arquivos)} cenários.\n")

    amostras = []
    total_lido = 0

    for i, path in enumerate(arquivos, 1):
        scenario = os.path.basename(os.path.dirname(os.path.dirname(path)))
        print(f"[{i:02d}/{len(arquivos)}] {scenario}", end=" ... ", flush=True)

        chunks_do_cenario = []
        for chunk in pd.read_csv(
            path,
            sep="\t",
            comment="#",
            header=None,
            names=COLS_TSV,
            low_memory=False,
            chunksize=CHUNK_SIZE,
        ):
            chunks_do_cenario.append(processar_chunk(chunk))

        df_cenario = pd.concat(chunks_do_cenario, ignore_index=True)
        df_cenario["scenario"] = scenario
        amostras.append(df_cenario)

        total_lido += len(df_cenario)
        print(f"{len(df_cenario):>10,} linhas amostradas")

    print(f"\nConcatenando {len(amostras)} cenários...")
    combined = pd.concat(amostras, ignore_index=True)

    print(f"Total final : {len(combined):,} linhas")
    print(f"Colunas     : {list(combined.columns)}")
    print(f"\nDistribuição de labels:\n{combined['label'].value_counts()}\n")
    print(f"Salvando em '{OUTPUT}'...")

    combined.to_csv(OUTPUT, index=False)
    size_mb = os.path.getsize(OUTPUT) / 1024 / 1024
    print(f"Arquivo salvo: {size_mb:.0f} MB")
    print("Concluído!")


if __name__ == "__main__":
    main()
