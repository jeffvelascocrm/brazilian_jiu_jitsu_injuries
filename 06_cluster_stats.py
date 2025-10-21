import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import shapiro, kruskal, chi2_contingency, fisher_exact
import statsmodels.api as sm
import scikit_posthocs as sp
import warnings
warnings.filterwarnings('ignore')

DATA_PATH = 'results/raw_dummies_clusters.csv'
RESULTS_PATH = 'results/'
FIG_PATH = os.path.join(RESULTS_PATH, 'figures')
os.makedirs(FIG_PATH, exist_ok=True)

df = pd.read_csv(DATA_PATH)
# Limpa espaços dos nomes das colunas
df.columns = [col.strip() for col in df.columns]

num_vars = ['Idade', 'Estatura', 'MC', 'Anos de JJ']
cluster_labels = sorted(df['cluster'].dropna().unique())

# Corrige separador decimal para ponto e converte para float
def fix_decimal(col):
    return col.astype(str).str.replace(',', '.').replace('', np.nan).astype(float)
for var in num_vars:
    if var in df.columns:
        df[var] = fix_decimal(df[var])

# 1. Normalidade e gráficos
normality_results = {}
for var in num_vars:
    normality_results[var] = {}
    for cl in cluster_labels:
        vals = df[df['cluster'] == cl][var].dropna()
        # Shapiro-Wilk
        if len(vals) >= 3:
            stat, p = shapiro(vals)
            if p < 0.05:
                conclusion = 'Not normal (p < 0.05)'
            else:
                conclusion = 'Normal (p ≥ 0.05)'
            normality_results[var][cl] = {'n': len(vals), 'shapiro_p': p, 'conclusion': conclusion}
        else:
            normality_results[var][cl] = {'n': len(vals), 'shapiro_p': None, 'conclusion': 'Sample too small for test'}
        # Histograma
        plt.figure(figsize=(5,3))
        sns.histplot(vals, kde=True)
        plt.title(f'{var} - Cluster {cl} (n={len(vals)})')
        plt.xlabel(var)
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_PATH, f'hist_{var}_cluster{cl}.png'))
        plt.close()
        # QQ-plot
        plt.figure(figsize=(5,3))
        sm.qqplot(vals, line='s')
        plt.title(f'QQ-plot {var} - Cluster {cl}')
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_PATH, f'qq_{var}_cluster{cl}.png'))
        plt.close()

# Salva resultados de normalidade
norm_rows = []
for var in num_vars:
    for cl in cluster_labels:
        res = normality_results[var][cl]
        norm_rows.append((var, cl, res['n'], res['shapiro_p'], res['conclusion']))
df_norm = pd.DataFrame(norm_rows, columns=['variable','cluster','n','shapiro_p','conclusion'])
df_norm.to_csv(os.path.join(RESULTS_PATH, 'normality_results.csv'), index=False)

# 2. Kruskal-Wallis
df_kruskal = []
kruskal_results = {}
for var in num_vars:
    vals_by_cluster = [df[df['cluster'] == cl][var].dropna() for cl in cluster_labels]
    stat, p = kruskal(*vals_by_cluster)
    kruskal_results[var] = {'stat': stat, 'p': p}
    df_kruskal.append((var, stat, p))
pd.DataFrame(df_kruskal, columns=['variable','kruskal_stat','kruskal_p']) \
    .to_csv(os.path.join(RESULTS_PATH, 'kruskal_results.csv'), index=False)

# 3. Post-hoc Dunn (se Kruskal-Wallis significativo)
for var in num_vars:
    if kruskal_results[var]['p'] < 0.05:
        data = df[[var, 'cluster']].dropna()
        dunn = sp.posthoc_dunn(data, val_col=var, group_col='cluster', p_adjust='bonferroni')
        dunn.to_csv(os.path.join(RESULTS_PATH, f'dunn_posthoc_{var}.csv'))

# 4. Categóricas/dummies: Qui-quadrado ou Fisher
cat_vars = [col for col in df.columns if (col.startswith('Local_') or col.startswith('Tipo_')) and df[col].nunique() == 2]
chi2_results = []
for var in cat_vars:
    table = pd.crosstab(df['cluster'], df[var])
    if table.shape[1] == 2:
        # Frequências baixas
        if (table.values < 5).sum() > 0:
            # Fisher para 2x2
            if table.shape[0] == 2:
                _, p = fisher_exact(table.values)
                chi2_results.append((var, 'Fisher', p))
            else:
                chi2_results.append((var, 'Chi2 (low expected)', None))
        else:
            chi2, p, _, _ = chi2_contingency(table)
            chi2_results.append((var, 'Chi2', p))
    else:
        chi2, p, _, _ = chi2_contingency(table)
        chi2_results.append((var, 'Chi2', p))

pd.DataFrame(chi2_results, columns=['variable','test','p']) \
    .to_csv(os.path.join(RESULTS_PATH, 'categorical_tests.csv'), index=False)

print('Análises de normalidade, Kruskal-Wallis, post-hoc e testes de lesão por cluster concluídas.')
print('Resultados e gráficos salvos em /results/ e /results/figures/')
