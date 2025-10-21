import os
import pandas as pd
import numpy as np

def explode_dummies(df, col):
    values = set()
    df[col] = df[col].fillna('')
    for items in df[col].astype(str):
        for item in items.split(','):
            i = item.strip()
            if i:
                values.add(i)
    for val in values:
        df[f'{col}_{val}'] = df[col].astype(str).apply(
            lambda x: int(val in [i.strip() for i in x.split(',')])
        )
    return df

RAW_PATH = 'data/raw/dados_respondentes_raw.csv'
CLUSTERS_PATH = 'results/clusters.csv'
OUTPUT_PATH = 'results/raw_dummies_clusters.csv'

# Load raw data and clusters, add respondent_id
raw = pd.read_csv(RAW_PATH)
raw = raw.reset_index().rename(columns={'index': 'respondent_id'})
clusters = pd.read_csv(CLUSTERS_PATH)

# Generate dummies for Local and Tipo
for c in ['Local', 'Tipo']:
    if c in raw.columns:
        raw = explode_dummies(raw, c)
    else:
        print(f'Aviso: coluna não encontrada para dummies: {c}')

# Merge using respondent_id
merged = pd.merge(raw, clusters[['respondent_id', 'cluster']], on='respondent_id', how='left')

# Save the new dataset
merged.to_csv(OUTPUT_PATH, index=False)
print(f'Dataset salvo em {OUTPUT_PATH}')
