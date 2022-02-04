from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 


class Evaluate(): 
    def __init__(self,**kwargs): 
        self.__dict__.update(kwargs)
        
    def get_mse(self,encoded_1,encoded_2): 
        loss = torch.nn.MSELoss()
        return loss(encoded_1,encoded_2)
    
    def make_pca(self,x,n_components = 2): 
        pca = PCA(n_components)
        pca_components = pd.DataFrame(pca.fit_transform(x),
              columns = [f'PC{n}' for n in range(1,n_components+1)]) 
        return pca_components

    def make_umap(self): 
        pass 
    
    def make_tsne(self,x,n_components = 2): 
        tsne = TSNE(n_components)
        tsne_components = pd.DataFrame(tsne.fit_transform(x),
              columns = [f'TSNE{n}' for n in range(1,n_components+1)]) 
        return tsne_components
    
    def compute_kmeans(self,x,n_clusters): 
        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(x)
        return kmeans.labels_
    
    def get_silhouette_coefficient(self,x,y):
        return silhouette_score(x,y)
    
    def compare_clustering(self,data,encoded_1,encoded_2,labels,n_kmeans_clusters = None):
        
        datasets = {"Original data":data, "Embeddings 1":encoded_1,"Embeddings 2":encoded_2}
        f,ax = plt.subplots(3,2,figsize = (20,20))
        ax = ax.flatten()
        it = 0 
        
        for name,dataset in datasets.items(): 
            dataset = dataset.detach().numpy()
            pca_components = self.make_pca(dataset)
            tsne_components = self.make_tsne(dataset)
            data_silhouette = self.get_silhouette_coefficient(dataset,labels)
            pca_silhouette = self.get_silhouette_coefficient(pca_components,labels)
            tsne_silhouette = self.get_silhouette_coefficient(tsne_components,labels)
            
            ax[it].scatter(data = pca_components, x = "PC1", y = "PC2",c = labels)
            ax[it+1].scatter(data = tsne_components, x = "TSNE1", y = "TSNE2", c = labels)
            ax[it].set(title = f" Dataset {name} - PCA components - pca silhouette = {pca_silhouette} -\n data silhouette = {data_silhouette} - ratio = {pca_silhouette/data_silhouette}")
            ax[it+1].set(title = f"Dataset {name} - TSNE components - tsne silhouette = {tsne_silhouette} -\n data silhouette = {data_silhouette} - ratio = {tsne_silhouette/data_silhouette}")
            it+=2
            