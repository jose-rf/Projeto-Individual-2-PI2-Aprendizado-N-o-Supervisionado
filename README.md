
# 📊 Projeto PI2 – Segmentação de Clientes com Aprendizado Não Supervisionado

Este projeto tem como objetivo aplicar técnicas de **Machine Learning Não Supervisionado** para realizar a **segmentação automática de clientes** com base em duas variáveis principais:

- **Idade**
- **Gasto Anual (R$)**

Para isso, foram utilizados dois métodos de clusterização:

- **K-Means Clustering**
- **Hierarchical Clustering (Ward Method)**

Os dados utilizados são **fictícios**, gerados com três segmentos naturais:

- 👦 Jovens — baixo gasto  
- 🧑 Adultos — gasto médio  
- 👴 Idosos — alto gasto  

Este repositório faz parte do **Projeto Individual (PI2)** da disciplina de Aprendizado de Máquina Não Supervisionado.

---

# 🚀 Como Rodar o Projeto

## **1. Requisitos**
Você precisa ter instalado:

- Python **3.8+**
- pip (gerenciador de pacotes)

## **2. Instalar dependências**
Execute no terminal:

```bash
pip install numpy pandas matplotlib scipy scikit-learn
```

## **3. Como executar cada script**

### ▶️ Rodar a versão do K-Means:
```bash
python kmeans_cliente.py
```

### ▶️ Rodar a versão do Hierarchical Clustering:
```bash
python hierarchical_cliente.py
```

## **4. Arquivos gerados automaticamente**
Ao executar os scripts, serão criados os seguintes gráficos:

- `kmeans_clusters.png`
- `dendrograma.png`
- `hierarchical_clusters.png`

E no terminal serão exibidos:

- Centroides (K-Means)
- Distribuição dos clusters
- Estatísticas por grupo
- Interpretação dos resultados

---

# 📁 Estrutura do Projeto

```
📦 projeto-segmentacao-clientes
 ┣ 📄 kmeans_cliente.py
 ┣ 📄 hierarchical_cliente.py
 ┣ 🖼 kmeans_clusters.png
 ┣ 🖼 dendrograma.png
 ┣ 🖼 hierarchical_clusters.png
 ┗ 📄 README.md
```

---

# 🧠 ETL – Extração, Tratamento e Preparação dos Dados

Os dois scripts realizam:

### ✔ Geração de dados fictícios em três clusters  
Usando distribuições normais com médias e desvios diferentes.

### ✔ Verificação de valores nulos  
Os dados são limpos na criação.

### ✔ Normalização  
É aplicada normalização Min-Max em:

- Idade  
- Gasto Anual  

Isto garante melhor desempenho nos algoritmos de clustering.

---

# 🤖 Modelos Utilizados

## 🔷 1. K-Means Clustering
O algoritmo K-Means:

- Segmenta pontos em *k* grupos definidos previamente
- Calcula centroides e atribui cada ponto ao centro mais próximo
- Funciona muito bem para clusters esféricos e bem separados

### 🟩 Resultados do K-Means
Os 3 grupos encontrados correspondem a:

| Cluster | Perfil | Idade Média | Gasto Médio | Ação Sugerida |
|--------|--------|-------------|-------------|----------------|
| 0 | Jovens | ~25 anos | ~R$ 2.000 | Promoções e entrada |
| 1 | Adultos | ~40 anos | ~R$ 5.000 | Upsell |
| 2 | Idosos | ~60 anos | ~R$ 8.000 | Fidelização VIP |

Gráfico gerado:

![K-Means Clusters](kmeans_clusters.png)

---

## 🔷 2. Hierarchical Clustering (Ward)
O método hierárquico:

- Não precisa de número inicial de clusters
- Analisa fusões progressivas entre pontos
- Permite visualizar a estrutura completa dos dados via dendrograma

### 🟩 Dendrograma
O dendrograma mostra 3 divisões naturais:

![Dendrograma](dendrograma.png)

### 🟩 Clusters após corte em 3 grupos
![Hierarchical Clusters](hierarchical_clusters.png)

---

# 📈 Comparação entre Algoritmos

| Critério | K-Means | Hierarchical Clustering |
|---------|---------|---------------------------|
| Escolha de clusters | Requer definir k antes | Definido visualmente no dendrograma |
| Performance | Muito rápido | Mais lento em grandes datasets |
| Interpretação | Simples e objetiva | Mais detalhada e visual |
| Outliers | Não lida bem | Pode tratar como fusões tardias |
| Escalabilidade | Excelente | Limitado a datasets menores |

**Conclusão:**  
✔ Para aplicações práticas → **K-Means**  
✔ Para análises exploratórias e estrutura dos dados → **Hierarchical**

---

# 📝 Conclusão Geral

Ambos os algoritmos segmentaram corretamente os três grupos naturais de clientes.  
No entanto:

- O **K-Means** é ideal para uso real em empresas, pois é rápido e eficiente.
- O **Hierarchical Clustering** é excelente para estudos iniciais e compreensão profunda da estrutura dos dados.

A combinação dos dois métodos oferece uma análise robusta, prática e interpretável.

---

# 👨‍💻 Autor
Projeto desenvolvido como parte do **PI2 – Aprendizado Não Supervisionado**.
