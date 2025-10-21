import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

RAW_PATH = 'data/raw/dados_respondentes_raw.csv'
PROC_PATH = 'data/processed/'
PROC_FILE = 'clustering_features.csv'
SAVE_PATH = os.path.join(PROC_PATH, PROC_FILE)
os.makedirs(PROC_PATH, exist_ok=True)

def clean_col(col):
    return col.strip().replace('  ', ' ')

df = pd.read_csv(RAW_PATH)
df.columns = [clean_col(c) for c in df.columns]

# Ajuste os nomes das variáveis obrigatórias conforme o CSV
obrigatorias = [
    'Sexo', 'Idade', 'Estatura', 'MC', 'Anos de JJ',
    'Grad. no JJ', 'Categoria de peso'  # Removido 'Treino semanal' e 'Duração'
]

# Exclui linhas com dados faltantes nas obrigatórias
for col in obrigatorias:
    if col not in df.columns:
        print(f'Aviso: coluna obrigatória não encontrada: {col}')
df = df.dropna(subset=[col for col in obrigatorias if col in df.columns])

# Padroniza separador decimal e converte para float nas numéricas
num_cols = [c for c in ['Idade', 'Estatura', 'MC', 'Anos de JJ'] if c in df.columns]
for col in num_cols:
    df[col] = df[col].astype(str).str.replace(',', '.').str.replace(' ', '').replace('', np.nan)
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Remover linhas com NaN restantes nas colunas numéricas
antes = len(df)
df = df.dropna(subset=num_cols)
depois = len(df)
if antes != depois:
    print(f'Removidas {antes - depois} linhas com valores numéricos ausentes.')

# Função para criar dummies binárias para colunas com múltiplos valores
# (ex: 'Ombro, Joelho')
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

# Criar dummies binárias para Local e Tipo (ajuste nomes conforme o CSV)
for c in ['Local', 'Tipo']:
    if c in df.columns:
        df = explode_dummies(df, c)
    else:
        print(f'Aviso: coluna não encontrada para dummies: {c}')

# Para coluna de lesão, se não teve lesão, os dummies ficam 0
if 'Lesao' in df.columns:
    lesao_cols = [col for col in df.columns if col.startswith('Local_') or col.startswith('Tipo_')]
    df['Lesao'] = df['Lesao'].fillna('Não').astype(str)
    for col in lesao_cols:
        df.loc[df['Lesao'].str.lower() == 'não', col] = 0

# Normalização z-score nas numéricas
scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

# Codificação categórica para k-prototypes
# - Sexo: LabelEncoder (0=Feminino, 1=Masculino)
if 'Sexo' in df.columns:
    le = LabelEncoder()
    df['Sexo'] = le.fit_transform(df['Sexo'].astype(str))

# Salvar CSV
cols_to_save = [col for col in obrigatorias if col in df.columns] + [col for col in df.columns if col.startswith('Local_') or col.startswith('Tipo_')]
df[cols_to_save].to_csv(SAVE_PATH, index=False)

print('Dataset para clusterização salvo em /data/processed/clustering_features.csv')
