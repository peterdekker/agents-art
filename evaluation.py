from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
from neupy import algorithms

import plot
from conf import ART_VIGILANCE, ART_LEARNING_RATE, N_CLUSTERS

def report_model(model, data_bin, words, model_name, n_clusters=None):
    score, clusters, data_std = evaluate_model(model, data_bin, model_name, n_clusters)
    print (f"Silhouette score: {score}")
    # Use standardized data for PCA plot
    plot.plot(data_std, words, clusters, model_name)
    # Give impression of clusters
    clusters_df = show_clusters(words, clusters).head(10)
    display(clusters_df)

def evaluate_model(model, data_bin, model_name, n_clusters=None):
    data_std = StandardScaler().fit_transform(data_bin)
    if model_name == "ART":
        clusters = model.predict(data_bin)
    elif model_name == "SOM-neupy":
        # Use standardized data
        model.train(data_std, epochs=100)
        predictions = model.predict(data_std)
        # TODO: can multiple neurons be winning? then argmax does not always make right decision
        clusters = np.argmax(predictions, axis=1)
    elif model_name == "SOM-minisom":
        if not n_clusters:
            raise ValueError("n_clusters should be given.")
        model.train(data_std, 1000)
        winner_coordinates = np.array([model.winner(x) for x in data_std]).T
        clusters = np.ravel_multi_index(winner_coordinates, (1,n_clusters))
    else:
        clusters = model.fit_predict(data_bin)
    score = silhouette_score(X=data_bin, labels=clusters, metric="hamming")
    return score, clusters, data_std

def show_clusters(words, clusters):
    df = pd.DataFrame()
    df["words"] = words
    df["clusters"] = [int(c) for c in clusters]
    piv_df = df.pivot(columns="clusters", values="words")
    sort_df = piv_df.apply(lambda x: pd.Series(x.dropna().values)).fillna("")
    return(sort_df)

def art_explore_parameters(data_bin, vigilances=None, learning_rates=None, n_clusters_settings = None):
    # If no parameter list given, evaluate only default value
    if not vigilances:
        vigilances=[ART_VIGILANCE]
    if not learning_rates:
        learning_rates = [ART_LEARNING_RATE]
    if not n_clusters_settings:
        n_clusters_settings = [N_CLUSTERS]
    assert isinstance(vigilances,list)
    assert isinstance(learning_rates,list)
    assert isinstance(n_clusters_settings,list)
    results_dict = {"vigilance":[], "learning_rate": [], "silhouette_score":[], "n_clusters":[]}
    for vig in vigilances:
        for lr in learning_rates:
            for nc in n_clusters_settings:
                artnet = algorithms.ART1(
                    step=lr,
                    rho=vig,
                    n_clusters=nc,
                    shuffle_data=False
                )
                score, _, _ = evaluate_model(artnet, data_bin, "ART")
                # Write new row to dict for dataframe
                results_dict["vigilance"].append(vig)
                results_dict["learning_rate"].append(lr)
                results_dict["n_clusters"].append(nc)
                results_dict["silhouette_score"].append(score)
    results_df = pd.DataFrame(results_dict)
    return results_df