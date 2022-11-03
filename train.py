x, y = generate_samples(100, 4)
k = K_Means(max_iter=100)
k.fit(x)
show_animation(x, y, k.get_history_centroids())
