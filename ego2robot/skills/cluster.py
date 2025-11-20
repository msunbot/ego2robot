"""Skill clustering."""
import numpy as np
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

class SkillClusterer:
    def __init__(self, config, n_clusters=10):
        self.config = config
        self.n_clusters = n_clusters
        self.kmeans = None
        self.tsne = None
        
    def fit(self, embeddings):
        """Fit K-means on embeddings."""
        print(f"Clustering {len(embeddings)} clips into {self.n_clusters} skills...")
        
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        cluster_ids = self.kmeans.fit_predict(embeddings)
        
        print(f"✓ Clustering complete")
        return cluster_ids
    
    def compute_tsne(self, embeddings):
        """Compute t-SNE for visualization."""
        print("Computing t-SNE (this may take a minute)...")
        self.tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings)-1))
        tsne_coords = self.tsne.fit_transform(embeddings)
        print("✓ t-SNE complete")
        return tsne_coords