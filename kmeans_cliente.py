# =============================================
# K-MEANS: Segmentação de Clientes
# Arquivo: kmeans_cliente.py
# =============================================

"""
1. DESCRIÇÃO DO PROBLEMA
Segmentar clientes de uma loja online com base em:
- Idade
- Gasto Anual (R$)

Dados fictícios com 3 grupos naturais:
- Jovens (baixo gasto)
- Adultos (gasto médio)
- Idosos (alto gasto)

Objetivo: Aplicar K-Means para descobrir esses grupos automaticamente.
"""

# --- Importação de bibliotecas ---
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

# --- 2. ETL: Geração e Preparação dos Dados ---
print("Gerando dados fictícios de clientes...")

np.random.seed(42)  # Reprodutibilidade
cluster1 = np.random.normal(loc=[25, 2000], scale=[5, 500], size=(100, 2))
cluster2 = np.random.normal(loc=[40, 5000], scale=[5, 1000], size=(100, 2))
cluster3 = np.random.normal(loc=[60, 8000], scale=[5, 1500], size=(100, 2))
data = np.vstack([cluster1, cluster2, cluster3])

df = pd.DataFrame(data, columns=['Idade', 'Gasto_Anual'])

print("Verificando valores nulos:")
print(df.isnull().sum())

# Normalização (Min-Max)
scaler = MinMaxScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

# --- 3. Aplicação do K-Means ---
print("\nAplicando K-Means (3 clusters)...")
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(df_scaled)

# Centroides no espaço original
centroids_original = scaler.inverse_transform(kmeans.cluster_centers_)
centroids_df = pd.DataFrame(centroids_original, columns=['Idade', 'Gasto_Anual'])

# --- 4. Visualização ---
plt.figure(figsize=(10, 6))
plt.scatter(df['Idade'], df['Gasto_Anual'], c=df['Cluster'], cmap='viridis', s=60, alpha=0.8, edgecolors='k')
plt.scatter(centroids_df['Idade'], centroids_df['Gasto_Anual'], 
            c='red', marker='X', s=300, label='Centroides', edgecolors='black', linewidth=2)
plt.title('K-Means: Segmentação de Clientes', fontsize=16, fontweight='bold')
plt.xlabel('Idade')
plt.ylabel('Gasto Anual (R$)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('kmeans_clusters.png', dpi=300, bbox_inches='tight')
plt.show()

# --- 5. Análise e Resultados ---
print("\n" + "="*60)
print("RESULTADOS DO K-MEANS")
print("="*60)
print(f"Inércia (WCSS): {kmeans.inertia_:.2f}")
print(f"\nCentroides (escala original):")
print(centroids_df.round(2))

print(f"\nDistribuição dos clientes por cluster:")
print(df['Cluster'].value_counts().sort_index())

print(f"\nMédias por cluster:")
print(df.groupby('Cluster')[['Idade', 'Gasto_Anual']].mean().round(2))

print("\nINTERPRETAÇÃO:")
print("• Cluster 0 → Jovens (~25 anos), baixo gasto (~R$2.000) → Promoções de entrada")
print("• Cluster 1 → Adultos (~40 anos), gasto médio (~R$5.000) → Upsell")
print("• Cluster 2 → Idosos (~60 anos), alto gasto (~R$8.000) → Fidelidade VIP")
print("="*60)

print("\nGráfico salvo como: kmeans_clusters.png")