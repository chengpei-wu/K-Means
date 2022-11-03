from k_means.utils import generate_samples, show_samples, show_animation
from k_means.k_means import K_Means

x, y = generate_samples(100, 4)
k = K_Means(max_iter=15)
k.fit(x)
show_animation(x, y, k.get_history_centroids())
