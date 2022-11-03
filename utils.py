from sklearn.datasets._samples_generator import make_blobs
import matplotlib.pyplot as plt


def generate_samples(n_samples, centers):
    return make_blobs(n_samples=n_samples, centers=centers, random_state=1234)


def show_samples(x, y, clusters_centroids):
    plt.title('samples')
    plt.scatter(x[:, 0], x[:, 1], color=[f'C{label}' for label in y])
    plt.scatter(clusters_centroids[:, 0], clusters_centroids[:, 1], c='black', marker='+')
    plt.xlabel('feature dimension 1')
    plt.ylabel('feature dimension 2')
    plt.show()


def show_animation(x, y, history_clusters_centroids):
    plt.title('centroids move process')
    fig = plt.figure(tight_layout=True)
    for point, label in zip(x, y):
        plt.scatter(point[0], point[1], color=f'C{label}')
    for cls in clusters_centroids:
        print(cls[0], cls[1])
        plt.scatter(cls[0], cls[1], c='black', marker='+')
    plt.xlabel('feature dimension 1')
    plt.ylabel('feature dimension 2')
    plt.show()
