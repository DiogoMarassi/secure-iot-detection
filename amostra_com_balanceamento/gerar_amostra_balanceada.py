"""
Gera amostra balanceada do dataset IoT-23.

Classificação dos cenários:
  Cenário A (< 16.265.399 flows): pega 100% dos dados
  Cenário B (≥ 16.265.399 flows): lógica de balanceamento por classe

Lógica Cenário B:
  - Classes raras (< 250.000 flows): pega 100%
  - Classes não raras (≥ 250.000 flows):
      1. Encontra a menor entre elas (min_count)
      2. Equaliza todas para min_count
      3. Distribui o orçamento restante (5% total - flows raros)
         igualmente entre as classes não raras

Saída: dataset_balanceado.csv
"""

import pandas as pd
import glob
import os

BASE         = "iot_23_datasets_small/opt/Malware-Project/BigDataset/IoTScenarios"
OUTPUT       = "dataset_balanceado.csv"
CHUNK_SIZE   = 500_000
RANDOM_SEED  = 42

LIMIAR_CENARIO     = 1_000_000  # separa Tipo A de Tipo B
LIMIAR_CLASSE_RARA = 250_000     # dentro do Tipo B, separa rara de não rara
SAMPLE_RATE        = 0.05        # 5% do total do cenário (aplicado nos Tipo B)

COLS_TSV = [
    "ts", "uid", "id.orig_h", "id.orig_p", "id.resp_h", "id.resp_p",
    "proto", "service", "duration", "orig_bytes", "resp_bytes",
    "conn_state", "local_orig", "local_resp", "missed_bytes", "history",
    "orig_pkts", "orig_ip_bytes", "resp_pkts", "resp_ip_bytes",
    "_last",   # contém: tunnel_parents   label   detailed-label (separados por 3 espaços)
]


# Funções auxiliares 

def ler_chunks(path):
    """Lê o arquivo em chunks, separando corretamente as últimas 3 colunas."""
    for chunk in pd.read_csv(
        path,
        sep="\t",
        comment="#",
        header=None,
        names=COLS_TSV,
        low_memory=False,
        chunksize=CHUNK_SIZE,
    ):
        split = chunk["_last"].str.split(r"\s{3}", expand=True)
        chunk["tunnel_parents"] = split[0]
        chunk["label"]          = split[1].str.strip()
        chunk["detailed-label"] = split[2].str.strip()
        chunk = chunk.drop(columns=["_last"])
        chunk = chunk.dropna(subset=["detailed-label"])
        yield chunk


def contar_classes(path):
    """
    Pass 1: percorre o arquivo em chunks contando flows por detailed-label.
    Retorna dict {classe: total_flows}.
    """
    counts = {}
    for chunk in ler_chunks(path):
        for label, count in chunk["detailed-label"].value_counts().items():
            counts[label] = counts.get(label, 0) + count
    return counts


def calcular_quotas(class_counts, total):
    """
    Calcula quantas linhas pegar de cada classe (Cenário B).
    Retorna dict {classe: quota_int}.
    """
    budget = int(total * SAMPLE_RATE)

    raras     = {k: v for k, v in class_counts.items() if v <  LIMIAR_CLASSE_RARA}
    nao_raras = {k: v for k, v in class_counts.items() if v >= LIMIAR_CLASSE_RARA}

    quotas = {}

    # Classes raras: 100%
    for label, count in raras.items():
        quotas[label] = count

    # Classes não raras: equalizar e distribuir o orçamento restante
    if nao_raras:
        rare_total  = sum(raras.values())
        min_count   = min(nao_raras.values())
        N           = len(nao_raras)
        remaining   = budget - rare_total
        per_class   = min(int(remaining / N), min_count)
        for label in nao_raras:
            quotas[label] = per_class

    return quotas, raras, nao_raras



def amostrar_cenario_a(path, scenario):
    """Cenário A: retorna 100% dos flows."""
    chunks = []
    for chunk in ler_chunks(path):
        chunk["scenario"] = scenario
        chunks.append(chunk)
    return pd.concat(chunks, ignore_index=True)


def amostrar_cenario_b(path, scenario, class_counts):
    """
    Cenário B: amostragem balanceada.
    Para cada classe calcula a fração = quota / count_real e aplica por chunk.
    """
    total  = sum(class_counts.values())
    quotas, raras, nao_raras = calcular_quotas(class_counts, total)

    # Fração de amostragem por classe
    fractions = {
        label: min(quota / class_counts[label], 1.0)
        for label, quota in quotas.items()
        if class_counts.get(label, 0) > 0
    }

    sampled = []
    for chunk in ler_chunks(path):
        rows = []
        for label, group in chunk.groupby("detailed-label"):
            frac = fractions.get(label)
            if frac is None:
                continue
            if frac >= 1.0:
                rows.append(group)
            else:
                rows.append(group.sample(frac=frac, random_state=RANDOM_SEED))
        if rows:
            sampled.append(pd.concat(rows))

    df = pd.concat(sampled, ignore_index=True)
    df["scenario"] = scenario
    return df, quotas, raras, nao_raras


#  Pipeline principal 

def main():
    arquivos = sorted(glob.glob(f"{BASE}/*/bro/conn.log.labeled"))
    print(f"Encontrados {len(arquivos)} cenários.\n")
    print("=" * 70)

    total_linhas = 0
    primeiro = True

    for i, path in enumerate(arquivos, 1):
        scenario = os.path.basename(os.path.dirname(os.path.dirname(path)))
        print(f"\n[{i:02d}/{len(arquivos)}] {scenario}")

        # Pass 1: contar classes 
        print("  Pass 1: contando classes...", end=" ", flush=True)
        class_counts = contar_classes(path)
        total = sum(class_counts.values())
        print(f"OK -> {total:,} flows | {len(class_counts)} classes")

        # Classificar e amostrar 
        if total < LIMIAR_CENARIO:
            print(f"  Tipo A -> pega 100% ({total:,} linhas)")
            df = amostrar_cenario_a(path, scenario)
            print(f"  Resultado: {len(df):,} linhas")

        else:
            print(f"  Tipo B -> Pass 2: amostrando...")
            df, quotas, raras, nao_raras = amostrar_cenario_b(
                path, scenario, class_counts
            )

            budget = int(total * SAMPLE_RATE)
            rare_total = sum(raras.values())

            print(f"  Orcamento (5%): {budget:,} flows")
            print(f"  Classes raras  ({len(raras)}): {rare_total:,} flows -> 100%")

            if nao_raras:
                min_count = min(nao_raras.values())
                per_class = quotas.get(next(iter(nao_raras)), 0)
                print(f"  Classes nao raras ({len(nao_raras)}): "
                      f"min={min_count:,} -> quota por classe={per_class:,}")

            print(f"  Resultado: {len(df):,} linhas")

        # Distribuição por classe no resultado
        dist = df["detailed-label"].value_counts()
        print("  Distribuicao:")
        for label, count in dist.items():
            original = class_counts.get(label, 0)
            pct = count / original * 100 if original else 0
            print(f"    {label:<45} {count:>8,}  ({pct:.1f}% do original)")

        # Escreve direto no CSV (sem acumular na RAM)
        df.to_csv(OUTPUT, mode="w" if primeiro else "a", index=False, header=primeiro)
        total_linhas += len(df)
        primeiro = False
        del df  # libera memória imediatamente

    # Resultado final 
    print("\n" + "=" * 70)
    size_mb = os.path.getsize(OUTPUT) / 1024 / 1024
    print(f"\nTotal final  : {total_linhas:,} linhas")
    print(f"Arquivo salvo: {OUTPUT} ({size_mb:.0f} MB)")
    print("Concluido!")


if __name__ == "__main__":
    main()
