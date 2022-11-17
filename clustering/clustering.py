from sklearn.metrics import silhouette_score
import pandas as pd
import gower
from numpy import unique
from numpy import where
from sklearn.cluster import *
from sklearn_extra.cluster import KMedoids, CommonNNClustering
import numpy as np
from sklearn.mixture import GaussianMixture
from clustering.survival_plot import log_rank_test
from collections import Counter

def cluster_score(labels):
    labels = list(labels)
    my_dict = dict(Counter(labels))
    score = 0
    for i in my_dict.keys():
        score += len(labels)/my_dict[i]
    
    return 1/score


def do_all_clustering(encoding_file_path, survival_file_path ):
    
    enc = encoding_file_path
    cols = enc.columns
    for col in cols:
        enc[col] = enc[col].astype(float)

    surv = pd.read_csv(survival_file_path, index_col=0)

    results = []

    X = np.array(enc)
    dist = gower.gower_matrix(enc)

    ## Kmeans
    try:
        for k in range(2, 7):
            kmeans = KMeans(n_clusters=k)
            kmeans.fit(enc)
            score = silhouette_score(enc, kmeans.labels_)
            survival_p_val = log_rank_test(enc.index, kmeans.labels_, surv)
            results.append(["KMEANS",k,score,survival_p_val,cluster_score(kmeans.labels_)])
    except:
        print("Error encountered in Kmeans clustering")

    ## PAM clustering
    try:
        for k in range(2, 7):
            pam = KMedoids(n_clusters=k,init='k-medoids++')
            pam.fit(dist)
            score = silhouette_score(dist, pam.labels_)
            survival_p_val = log_rank_test(enc.index, pam.labels_, surv)
            results.append(["PAM",k,score,survival_p_val,cluster_score(pam.labels_)])
    except:
        print("Error encountered in PAM clustering")

    ## Common NN
    try:
        clustering = CommonNNClustering(eps=0.005, min_samples=0)
        clustering.fit(enc)
        score = silhouette_score(enc, clustering.labels_)
        survival_p_val = log_rank_test(enc.index, clustering.labels_, surv)
        k = len(unique(clustering.labels_))
        results.append(["COMMONNN",k,score,survival_p_val,cluster_score(clustering.labels_)])
    except:
        print("Error encountered in COMMONNN clustering")

    ## Affinity Propagation
    try:
        model = AffinityPropagation(damping=0.9)
        model.fit(dist)
        yhat = model.predict(dist)
        score = silhouette_score(dist, yhat)
        survival_p_val = log_rank_test(enc.index, yhat, surv)
        k = len(unique(yhat))
        results.append(["AFFINITY_PROPAGATION",k,score,survival_p_val,cluster_score(yhat)])
    except:
        print("Error encountered in AFFINITY_PROPAGATION clustering")
   
   ## Agglomerative clustering
    try:
        for k in range(2, 7):
            model = AgglomerativeClustering(n_clusters=k)
            yhat = model.fit_predict(X)
            score = silhouette_score(enc, yhat)
            survival_p_val = log_rank_test(enc.index, yhat, surv)
            results.append(["AGGLOMERATIVE_CLUSTERING",k,score,survival_p_val,cluster_score(yhat)])
    except:
        print("Error encountered in AGGLOMERATIVE_CLUSTERING clustering")

    ## Birch
    try:
        for k in range(2, 7):
            clust = Birch(n_clusters=k, threshold=0.1)
            clust.fit(dist)
            score = silhouette_score(dist, clust.labels_)
            survival_p_val = log_rank_test(enc.index, clust.labels_, surv)
            results.append(["BRICH",k,score,survival_p_val,cluster_score(clust.labels_)])
    except:
        print("Error encountered in BRICH clustering")

    ## DBscan
    try:
        model = DBSCAN(eps=0.01, min_samples=10)
        yhat = model.fit_predict(X)
        score = silhouette_score(enc, yhat)
        survival_p_val = log_rank_test(enc.index, yhat, surv)
        k = len(unique(yhat))
        results.append(["DBSCAN",k,score,survival_p_val,cluster_score(yhat)])
    except:
        print("Error encountered in DBSCAN clustering")

    ## Meanshift
    try:
        model = MeanShift()
        yhat = model.fit_predict(X)
        score = silhouette_score(enc, yhat)
        survival_p_val = log_rank_test(enc.index, yhat, surv)
        k = len(unique(yhat))
        results.append(["MEANSHIFT",k,score,survival_p_val,cluster_score(yhat)])
    except:
        print("Error encountered in MEANSHIFT clustering")

    ## Spectral Clustering
    try:
        for k in range(2, 7):
            clust = SpectralClustering(n_clusters=k)
            clust.fit(X)
            score = silhouette_score(enc, clust.labels_)
            survival_p_val = log_rank_test(enc.index, clust.labels_, surv)
            results.append(["SPECTRAL_CLUSTERING",k,score,survival_p_val,cluster_score(clust.labels_)])
    except:
        print("Error encountered in SPECTRAL_CLUSTERING clustering")

    ## Gaussian Mixture
    try:
        for k in range(2, 7):
            model = GaussianMixture(n_components=k)
            yhat = model.fit_predict(X)
            score = silhouette_score(enc, yhat)
            survival_p_val = log_rank_test(enc.index, yhat, surv)
            results.append(["GAUSSIAN_MIXTURE",k,score,survival_p_val,cluster_score(yhat)])
    except:
        print("Error encountered in GAUSSIAN_MIXTURE clustering")


    return results
