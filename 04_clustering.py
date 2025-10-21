import os
import pandas as pd
import numpy as np
from kmodes.kprototypes import KPrototypes
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import unidecode

# Paths
PROC_PATH = 'data/processed/'
RESULTS_PATH = 'results/'
DATA_PATH = os.path.join(PROC_PATH, 'clustering_features.csv')
os.makedirs(RESULTS_PATH, exist_ok=True)
os.makedirs(os.path.join(RESULTS_PATH, 'figures'), exist_ok=True)

df = pd.read_csv(DATA_PATH)
df = df.reset_index().rename(columns={'index': 'respondent_id'})  # Add unique ID

features = ['Sexo', 'Idade', 'Estatura', 'MC', 'Anos de JJ', 'Grad. no JJ', 'Categoria de peso']
lesion_cols = [col for col in df.columns if (col.startswith('Local_') or col.startswith('Tipo_')) and pd.api.types.is_numeric_dtype(df[col])]
controle_mask = (df[lesion_cols].sum(axis=1) == 0)

# Assign clusters: 0 = control, 1-3 = injured clusters
clusters_all = np.full(len(df), -1)
clusters_all[controle_mask] = 0

# Cluster injured athletes
cluster_mask = ~controle_mask
X = df.loc[cluster_mask, features].copy()
cat_cols = [i for i, col in enumerate(features) if col in ['Sexo', 'Grad. no JJ', 'Categoria de peso']]
kproto = KPrototypes(n_clusters=3, init='Cao', n_init=10, verbose=0)
clusters_injured = kproto.fit_predict(X.values, categorical=cat_cols)
clusters_all[cluster_mask] = clusters_injured + 1  # clusters 1,2,3
df['cluster'] = clusters_all

# Save clusters.csv (with respondent_id)
output_file = os.path.join(RESULTS_PATH, 'clusters.csv')
df.to_csv(output_file, index=False)

# Cluster summary with centroids
cluster_summaries = []
for cl in sorted(df['cluster'].unique()):
    sub = df[df['cluster']==cl][features]
    summary_dict = {'cluster': cl, 'n_athletes': len(sub)}
    for f in features:
        if sub[f].dtype == object:
            summary_dict[f+'_centroid'] = sub[f].mode(dropna=True).values[0] if not sub[f].mode(dropna=True).empty else np.nan
        else:
            summary_dict[f+'_centroid'] = sub[f].mean()
    cluster_summaries.append(summary_dict)
summary_df = pd.DataFrame(cluster_summaries)
summary_df.to_csv(os.path.join(RESULTS_PATH, 'cluster_summary.csv'), index=False)

# PCA 3D visualization (all clusters)
pca_df = df[features].copy()
for col in ['Sexo','Grad. no JJ','Categoria de peso']:
    pca_df[col] = pca_df[col].astype('category').cat.codes
pca3d = PCA(n_components=3).fit_transform(pca_df)
pca3d_df = pd.DataFrame(pca3d, columns=['PC1','PC2','PC3'])
pca3d_df['cluster'] = df['cluster']
colors_3d = {0:'#888888', 1:'#FF0000', 2:'#008000', 3:'#0000FF'}
fig = plt.figure(figsize=(9,8))
ax = fig.add_subplot(111, projection='3d')
for clust in sorted(pca3d_df['cluster'].unique()):
    ix = pca3d_df['cluster']==clust
    ax.scatter(
        pca3d_df.loc[ix, 'PC1'], pca3d_df.loc[ix, 'PC2'], pca3d_df.loc[ix, 'PC3'],
        color=colors_3d[int(clust)], label=f'Cluster {int(clust)}', s=70, edgecolors='k')
ax.legend(title='Cluster', loc='best')
ax.set_title('Clusters (including control) - PCA 3D')
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_PATH, 'figures', 'clusters_pca3d_all.png'))
plt.close()

# PCA 3D visualization (injured clusters only)
pca3d_inj = pca3d_df[pca3d_df['cluster']>0]
fig = plt.figure(figsize=(9,8))
ax = fig.add_subplot(111, projection='3d')
for clust in sorted(pca3d_inj['cluster'].unique()):
    ix = pca3d_inj['cluster']==clust
    ax.scatter(
        pca3d_inj.loc[ix, 'PC1'], pca3d_inj.loc[ix, 'PC2'], pca3d_inj.loc[ix, 'PC3'],
        color=colors_3d[int(clust)], label=f'Cluster {int(clust)}', s=70, edgecolors='k')
ax.legend(title='Cluster', loc='best')
ax.set_title('Injured Clusters Only - PCA 3D')
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_PATH, 'figures', 'clusters_pca3d_injured.png'))
plt.close()

print(summary_df)
print('Salvo: clusters.csv, cluster_summary.csv, clusters_pca3d_all.png, clusters_pca3d_injured.png em /results/')
