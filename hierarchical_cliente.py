# =============================================
# HIERARCHICAL CLUSTERING: Segmentação de Clientes
# Arquivo: hierarchical_cliente.py
# =============================================

"""
1. DESCRIÇÃO DO PROBLEMA
Mesmos dados do K-Means: Idade e Gasto Anual.
Agora usamos Hierarchical Clustering (método Ward) para:
- Visualizar hierarquia com dendrograma
- Identificar clusters em diferentes níveis
"""

# --- Importação de bibliotecas ---
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.preprocessing import MinMaxScaler

# --- 2. ETL: Geração e Preparação ---
print("Gerando dados fictícios (mesmos do K-Means)...")

np.random.seed(42)
cluster1 = np.random.normal(loc=[25, 2000], scale=[5, 500], size=(100, 2))
cluster2 = np.random.normal(loc=[40, 5000], scale=[5, 1000], size=(100, 2))
cluster3 = np.random.normal(loc=[60, 8000], scale=[5, 1500], size=(100, 2))
data = np.vstack([cluster1, cluster2, cluster3])

df = pd.DataFrame(data, columns=['Idade', 'Gasto_Anual'])
print("Valores nulos:")
print(df.isnull().sum())

# Normalização
scaler = MinMaxScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

# --- 3. Hierarchical Clustering ---
print("\nComputando linkage (método Ward)...")
Z = linkage(df_scaled, method='ward')

# Cortar em 3 clusters
df['Cluster'] = fcluster(Z, t=3, criterion='maxclust') - 1

# --- 4. Visualizações ---
# Dendrograma
plt.figure(figsize=(12, 6))
dendrogram(Z, truncate_mode='level', p=3, leaf_rotation=90, color_threshold=6)
plt.axhline(y=6, color='red', linestyle='--', label='Corte em 3 clusters')
plt.title('Dendrograma - Hierarchical Clustering (Ward)', fontsize=16, fontweight='bold')
plt.xlabel('Clientes (ou tamanho do cluster)')
plt.ylabel('Distância (Ward)')
plt.legend()
plt.tight_layout()
plt.savefig('dendrograma.png', dpi=300, bbox_inches='tight')
plt.show()

# Scatter com clusters
plt.figure(figsize=(10, 6))
plt.scatter(df['Idade'], df['Gasto_Anual'], c=df['Cluster'], cmap='plasma', s=60, alpha=0.8, edgecolors='k')
plt.title('Hierarchical Clustering: 3 Grupos de Clientes', fontsize=16, fontweight='bold')
plt.xlabel('Idade')
plt.ylabel('Gasto Anual (R$)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('hierarchical_clusters.png', dpi=300, bbox_inches='tight')
plt.show()

# --- 5. Análise ---
print("\n" + "="*60)
print("RESULTADOS DO HIERARCHICAL CLUSTERING")
print("="*60)
print("Distribuição por cluster:")
print(df['Cluster'].value_counts().sort_index())

print("\nMédias por cluster:")
print(df.groupby('Cluster')[['Idade', 'Gasto_Anual']].mean().round(2))

print("\nINTERPRETAÇÃO:")
print("• Resultados muito similares ao K-Means")
print("• Dendrograma mostra fusão progressiva dos pontos")
print("• Vantagem: permite escolher número de clusters sem reexecutar")
print("• Ideal para dados com estrutura hierárquica natural")
print("="*60)

print("\nGráficos salvos:")
print("  → dendrograma.png")
print("  → hierarchical_clusters.png")