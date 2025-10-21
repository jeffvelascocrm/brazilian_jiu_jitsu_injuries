import pandas as pd
import numpy as np

DATA_PATH = 'results/raw_dummies_clusters.csv'
OUTPUT_PATH = 'results/cluster_report.csv'

df = pd.read_csv(DATA_PATH)
df.columns = [col.strip() for col in df.columns]

num_vars = ['Idade', 'Estatura', 'MC', 'Anos de JJ']
cat_vars = ['Sexo', 'Grad. no JJ', 'Categoria de peso']
lesion_vars = [col for col in df.columns if col.startswith('Local_') or col.startswith('Tipo_')]

# Corrige tipo das dummies de lesão para inteiro
for var in lesion_vars:
    if var in df.columns:
        df[var] = pd.to_numeric(df[var], errors='coerce').fillna(0).astype(int)

clusters = sorted(df['cluster'].dropna().unique())
report_rows = []

for cl in clusters:
    sub = df[df['cluster'] == cl]
    row = {'Cluster': cl, 'n': len(sub)}
    # Numéricas
    for var in num_vars:
        # Corrige separador decimal e converte para float
        vals = sub[var].astype(str).str.replace(',', '.').replace('', np.nan).astype(float)
        row[f'{var}_mean'] = vals.mean()
        row[f'{var}_median'] = vals.median()
        row[f'{var}_std'] = vals.std()
    # Categóricas
    for var in cat_vars:
        if var in sub.columns:
            row[f'{var}_mode'] = sub[var].mode(dropna=True).values[0] if not sub[var].mode(dropna=True).empty else np.nan
    # Lesões (apenas clusters 1,2,3)
    if cl in [1,2,3]:
        for var in lesion_vars:
            if var in sub.columns:
                row[f'{var}_prev'] = 100 * sub[var].sum() / len(sub) if len(sub) > 0 else 0
    report_rows.append(row)

# Gera DataFrame e salva
report_df = pd.DataFrame(report_rows)
report_df.to_csv(OUTPUT_PATH, index=False)
print(f'Relatório salvo em {OUTPUT_PATH}')
