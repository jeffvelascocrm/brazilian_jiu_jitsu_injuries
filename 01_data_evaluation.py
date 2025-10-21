import os
import pandas as pd
import numpy as np

RAW_DATA_PATH = 'data/raw/dados_respondentes_raw.csv'
PROCESSED_PATH = 'data/processed/'
RESULTS_PATH = 'results/'
FIGURES_PATH = os.path.join(RESULTS_PATH, 'figures/')

for path in [PROCESSED_PATH, RESULTS_PATH, FIGURES_PATH]:
    os.makedirs(path, exist_ok=True)

# Carregamento de dados
try:
    df = pd.read_csv(RAW_DATA_PATH)
except FileNotFoundError:
    print(f'Arquivo não encontrado: {RAW_DATA_PATH}')
    exit(1)

# Resumo completo para todas as colunas
full_summary = df.describe(include='all').transpose()
full_summary.to_csv(os.path.join(RESULTS_PATH, 'full_summary.csv'))

# Frequência absoluta das variáveis categóricas
categorical_cols = df.select_dtypes(include=['object', 'category']).columns
with open(os.path.join(RESULTS_PATH, 'categorical_frequencies.csv'), 'w', encoding='utf-8') as f:
    f.write('variable,value,frequency\n')
    for col in categorical_cols:
        freq = df[col].value_counts(dropna=False)
        for val, count in freq.items():
            f.write(f'{col},{val},{count}\n')

# Resumo dos dados faltantes
missing_summary = df.isnull().sum()
missing_summary.to_csv(os.path.join(RESULTS_PATH, 'missing_summary.csv'), header=['missing_count'])

# Outlier report numérico
numeric_cols = df.select_dtypes(include=[np.number]).columns
outlier_report = []
for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outliers = df[(df[col] < lower) | (df[col] > upper)][col]
    if not outliers.empty:
        outlier_report.append(f'Coluna: {col}\nOutliers: {outliers.values}\n')
    else:
        outlier_report.append(f'Coluna: {col}\nOutliers: Nenhum\n')
with open(os.path.join(RESULTS_PATH, 'outlier_report.txt'), 'w', encoding='utf-8') as f:
    f.writelines(outlier_report)

print('Resumo completo salvo em results/full_summary.csv')
print('Frequências categóricas em results/categorical_frequencies.csv')
print('Relatórios de missing e outliers atualizados.')
