import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from kmodes.kprototypes import KPrototypes
from sklearn.metrics import silhouette_score

PROC_PATH = 'data/processed/'
RESULTS_PATH = 'results/'
os.makedirs(RESULTS_PATH, exist_ok=True)
os.makedirs(os.path.join(RESULTS_PATH, 'figures'), exist_ok=True)

DATA_PATH = os.path.join(PROC_PATH, 'clustering_features.csv')
df = pd.read_csv(DATA_PATH)

features = ['Sexo', 'Idade', 'Estatura', 'MC', 'Anos de JJ', 'Grad. no JJ', 'Categoria de peso']
# Seleciona apenas colunas binárias de lesão (int64)
lesion_cols = [col for col in df.columns if (col.startswith('Local_') or col.startswith('Tipo_')) and df[col].dtype in [np.int64, np.int32, 'int64']]

# Grupo controle: todos os registros onde todas as colunas de lesão binária são zero
controle_mask = (df[lesion_cols].sum(axis=1) == 0)
controle = df[controle_mask]
data = df[~controle_mask]

# Apenas as features para clusterização
X = data[features].copy()

# Identifica colunas categóricas para k-prototypes
cat_cols = [i for i, col in enumerate(X.columns) if col in ['Sexo', 'Grad. no JJ', 'Categoria de peso']]

elbow_scores = []
silhouette_scores = []
k_range = range(2, 9)
for k in k_range:
    kproto = KPrototypes(n_clusters=k, init='Cao', n_init=5, verbose=0)
    clusters = kproto.fit_predict(X.values, categorical=cat_cols)
    cost = kproto.cost_
    elbow_scores.append(cost)
    # Para calcular o silhouette, é preciso transformar categóricas em numéricas
    tmp_X = X.copy()
    for col in ['Sexo', 'Grad. no JJ', 'Categoria de peso']:
        tmp_X[col] = tmp_X[col].astype('category').cat.codes
    try:
        sil_score = silhouette_score(tmp_X, clusters)
    except:
        sil_score = np.nan
    silhouette_scores.append(sil_score)

elbow_df = pd.DataFrame({'k':list(k_range), 'cost':elbow_scores})
silhouette_df = pd.DataFrame({'k':list(k_range), 'silhouette':silhouette_scores})
elbow_df.to_csv(os.path.join(RESULTS_PATH, 'elbow_scores.csv'), index=False)
silhouette_df.to_csv(os.path.join(RESULTS_PATH, 'silhouette_scores.csv'), index=False)

plt.figure(figsize=(8,5))
plt.plot(list(k_range), elbow_scores, marker='o')
plt.title('Elbow Method - Custos por k')
plt.xlabel('Número de clusters (k)')
plt.ylabel('Custo do k-prototypes')
plt.xticks(list(k_range))
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_PATH, 'figures', 'elbow_plot.png'))
plt.close()

plt.figure(figsize=(8,5))
plt.plot(list(k_range), silhouette_scores, marker='o', color='orange')
plt.title('Silhouette Score por k')
plt.xlabel('Número de clusters (k)')
plt.ylabel('Silhouette Score')
plt.xticks(list(k_range))
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_PATH, 'figures', 'silhouette_plot.png'))
plt.close()

print('Resultados salvos em /results/ para decisão de k.')
