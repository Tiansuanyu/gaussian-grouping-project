# Feature Cluster Summary

| Scene | Algorithm | Iteration | Points | Dim | Clusters | Silhouette | Silhouette samples | PCA var (3D) | Plot | Status |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| ramen | kmeans | 30000 | 1114069 | 16 | 16 | 0.2478 | 4992 | 0.375959 | analysis/feature_cluster_results/ramen/iteration_30000/kmeans/scatter_3d.png | ok |
| ramen | hdbscan | 30000 | 1114069 | 16 | 3 | 0.5425 | 2932 | 0.375959 | analysis/feature_cluster_results/ramen/iteration_30000/hdbscan/scatter_3d.png | ok |
| figurines | kmeans | 30000 | 280344 | 16 | 16 | 0.2445 | 4992 | 0.381563 | analysis/feature_cluster_results/figurines/iteration_30000/kmeans/scatter_3d.png | ok |
| figurines | hdbscan | 30000 | 280344 | 16 | 2 | 0.4953 | 3425 | 0.381563 | analysis/feature_cluster_results/figurines/iteration_30000/hdbscan/scatter_3d.png | ok |
| teatime | kmeans | 30000 | 279705 | 16 | 16 | 0.2383 | 4992 | 0.446658 | analysis/feature_cluster_results/teatime/iteration_30000/kmeans/scatter_3d.png | ok |
| teatime | hdbscan | 30000 | 279705 | 16 | 2 | 0.3593 | 3016 | 0.446658 | analysis/feature_cluster_results/teatime/iteration_30000/hdbscan/scatter_3d.png | ok |

Interpretation: KMeans is the all-point, fixed-16-cluster diagnostic and is the primary score for 3D separability. HDBSCAN is fit on a sampled subset when requested and its silhouette is computed only on non-noise dense clusters, so those scores can be higher even when the full feature cloud visibly overlaps.
