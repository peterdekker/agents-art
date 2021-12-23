from neupy import algorithms


import data
import plot


def main():
    data_bin, words = data.load_vectors("binvectors256.vec", n_words=50)
    artnet = algorithms.ART1(
        step=0.1,
        rho=0.1,
        n_clusters=10,
        shuffle_data=False
    )
    clusters_art = artnet.predict(data_bin)
    plot.plot(data_bin, words, clusters_art)

if __name__ == "__main__":
    main()