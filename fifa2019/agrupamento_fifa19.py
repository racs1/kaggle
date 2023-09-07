# IMPORTAÇÃO DAS BIBLIOTECAS
#
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.ensemble import RandomForestClassifier
from collections import Counter
#

# CARREGAMENTO DA BASE DE DADOS 
#
df = pd.read_csv("C:/workspace/kaggle-fifa2019/data.csv")
df.shape
df.head()
# visualizando o índice das colunas para slice dos dados desejados
for i, column in enumerate(df.columns):
    print(i, column)
# col21(position), col26(height), col27(weight), col54(crossing) a col82(slidingtackle)
columns = [21,26,27]
columns += range(54,83)
print(columns)
# selecionando o subconjunto que iremos trabalhar
df=df.iloc[:,columns]
df.head()
# valores faltantes
df.isna().sum()
# valores faltantes (dropna - pequeno impacto de 0,33%)
((len(df)-len(df.dropna()))/(len(df)))*100
df = df.dropna()
df.shape
df.isna().sum()
#

# INSPEÇÃO DE ESTATÍSTICAS
#
def hist_boxplot(feature):
    fig, ax = plt.subplots(1,2)
    ax[0].hist(feature)
    ax[1].boxplot(feature)
df_describe = df.describe()
hist_boxplot(df_describe.loc['min'])
hist_boxplot(df_describe.loc['max'])
hist_boxplot(df_describe.loc['mean'])

# TRATAMENTO DE ATRIBUTOS 
#
df.dtypes
# localidando atributos diferentes de float64 e int64
df.dtypes[(x not in ['int64', 'float64'] for x in df.dtypes)]
# tratando altura dos jogadores - transformando 5'7 (5 pés e 7 polegadas) em [5,7] (lista)
df['Height']=df['Height'].str.split('\'')
df['Height']
df.dtypes
# conversão p/ cm: 30.48 * pés + 2.54 * polegadas
df['Height']= [30.48*int(item[0]) + 2.54*int(item[1]) for item in df['Height']]
df['Height']
df.dtypes
# tratando peso dos jogadores 
df['Weight']
#conversão de libras para Kg: 1 lbs = 0,453592 Kg
df['Weight'] = df['Weight'].str.split('l')
df['Weight'].head()
df['Weight']=[0.45*int(item[0]) for item in df['Weight']]
df['Weight']
df.dtypes
hist_boxplot(df['Weight'])
#

# PRÉ-PROCESSAMENTO DA BASE DE DADOS PARA AGRUPAMENTO
#
position = np.array(df['Position'])
print(position)
np.unique(position, return_counts=True)
df=df.drop(['Position'], axis = 1)
df.head()
# normalizando os dados 
scaler = MinMaxScaler()
df_train = scaler.fit_transform(df)
type(df_train) 
#

# AGRUPAMENTO K-MEANS
#
# escolha do número de clusters com WCSS
wcss = []
K = range(1,12) #11 jogadores
for k in K:
    km = KMeans(n_clusters = k)
    km = km.fit(df_train)
    wcss.append(km.inertia_)
wcss
plt.plot(K, wcss, 'bx-')
plt.xlabel('k')
plt.ylabel('WCSS')
plt.title('Elbow Method for Optimal k');

# redução de dimensionalidade com PCA
pca = PCA(n_components=2)
df_pca = pca.fit_transform(df_train)
df_pca
df_train.shape
df_pca.shape
pca.explained_variance_ratio_
exp_var = [round(i,1) for i in pca.explained_variance_ratio_ * 100]
exp_var
