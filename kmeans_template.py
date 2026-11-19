"""Question 2: K-means (30 points), adapted from Assignment 4, Question 2.

Implement the two update functions from scratch, then complete the experiment.
"""

import argparse
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

K_VALUES = (2, 3, 4, 6, 10)


def update_assignments(data, centroids):
    """Return one cluster index per row, using nearest-centroid distance.

    Q2(a): the two update functions together are worth 15 points.
    data has shape (n, d); centroids has shape (k, d).
    The result should be an integer array with shape (n,).
    Resolve equal-distance ties consistently and document your rule.
    """
    # TODO: Compute distances and assign each point to a centroid.
    # This array will store what point belongs to what cluster. Assignment[i] = k means pt i belongs to cluster k. This array will be returned in the end
    assignment = np.zeros(data.shape[0], dtype=int)

    # iterate over every point
    for ptIdx in range(0, data.shape[0]):
        # predefine varibable to avoid recalculations
        closestCentroidIdx = 0
        # init minDist with first centroid
        minDist = np.linalg.norm(centroids[0] - data[ptIdx])
        # in this loop we just iterate over every centroid and see which one is closest and store it in minDist
        for centeroidIdx in range(1, centroids.shape[0]):
            currDist = np.linalg.norm(centroids[centeroidIdx]- data[ptIdx])
            if(currDist < minDist):
                minDist = currDist
                closestCentroidIdx = centeroidIdx

            # if two centroids are at same distance from a point, select the one with lowest idx. Done implicitly
        # Assign closest centroid to that pt
        assignment[ptIdx] = closestCentroidIdx
    return assignment







def update_centroids(data, centroids, assignments):
    """Return an updated (k, d) centroid array based on cluster assignments.

    Document how you handle an empty cluster. One valid choice is to retain
    its previous centroid. Avoid modifying the input centroid array in place.
    """
    # create newCentroids array to avoid modifying old one
    newCentroids = np.zeros(centroids.shape, dtype=float)
    ptCount = np.zeros(centroids.shape[0], dtype=int)

    # find ptCount count and pt sum for all clusters to calculate new average later on
    for i in range(0, assignments.shape[0]):
        newCentroids[assignments[i]] += data[i]
        ptCount[assignments[i]] += 1

    # calculate the new centroids by dividing cluster sum by cluster count
    for i in range(0, newCentroids.shape[0]):
        if(ptCount[i] > 0):
            newCentroids[i] /= ptCount[i]
        else: #empty clusters preserve their original mean
            newCentroids[i] = centroids[i]
    return newCentroids


# used for determining how good cluster assingments are
def calc_inertia(data, centroids, assignments):
    """Return the k-means objective: the sum of squared distances from each
    point to its assigned centroid (within-cluster sum of squares)."""
    return ((data - centroids[assignments]) ** 2).sum()




def kmeans(data, centroids, max_iterations=100):
    """Assignment 4's provided alternating-update loop, with a final assignment."""
    for _ in range(max_iterations):
        assignments = update_assignments(data, centroids)
        centroids = update_centroids(data, centroids, assignments)
    assignments = update_assignments(data, centroids)
    return centroids, assignments


def make_toy_data():
    """Generate the exact make_blobs configuration from Assignment 4."""
    from sklearn.datasets import make_blobs
    return make_blobs(n_samples=4000, centers=4, cluster_std=0.60, random_state=0)


def get_initial_clusters(k, data, rng):
    """Sample k rows with replacement, as in the source's randint initializer.

    This adaptation accepts a seeded NumPy generator for reproducibility.
    Repeated initial rows are possible; your empty-cluster policy must cope.
    """
    return data[rng.integers(0, data.shape[0], size=k)].copy()



def plot_kmeans(data, centroids, assignments, name, output_dir):
    """Save a scatter plot of clustered data and centroids to output_dir/name.

    Uses the first two features. `name` is a filename such as "kmeans_k3.png".
    The output directory is created if it doesn't exist.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    k = centroids.shape[0]
    cmap = plt.get_cmap("tab10" if k <= 10 else "tab20")

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(data[:, 0], data[:, 1], c=assignments, cmap=cmap,
               vmin=0, vmax=cmap.N - 1, s=20, alpha=0.7)
    ax.scatter(centroids[:, 0], centroids[:, 1],
               c=np.arange(k), cmap=cmap, vmin=0, vmax=cmap.N - 1,
               marker="X", s=250, edgecolors="black", linewidths=1.5,
               label="Centroids")

    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.set_title(f"K-means clustering (k={k})")
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_dir / name, dpi=150)
    plt.close(fig)


def run_experiments(output_dir, seed=0, max_iterations=100):
    """Q2(b), 15 points: run all five k values and save cluster plots.

    Use the four-component truth labels only for a reference plot. K-means
    must cluster the features without access to those labels.
    """
    # TODO: Load toy data, create a seeded RNG, and initialize centroids.
    # TODO: Run kmeans for every value in K_VALUES.
    # TODO: Plot each set of cluster assignments and its final centroids.
    # TODO: Save plots under output_dir and discuss the effect of k.

    
    # update_assignments: Ties: a point equidistant from several centroids goes to the lowest-index centroid (strict '<' keeps the earlier one).
    # update_centroids: Empty clusters: a cluster with no assigned points keeps its previous centroid.
    
    data, truth = make_toy_data()
    rng = np.random.default_rng(seed)

    true_centers = np.array([data[truth == j].mean(axis=0) for j in range(4)])
    plot_kmeans(data, true_centers, truth, "truth.png", output_dir)

    kmeans_times = 10 
    for k in K_VALUES:
        results = []
        for _ in range(0, kmeans_times): # run multiple times to try avoid getting stuck in a local inertia minimum
            centroids = np.zeros((k, data.shape[1])) 
            seen = set()
            # chose k random points to be initial centroids
            for i in range(0, k):
                x = rng.integers(0, data.shape[0])
                while x in seen:
                    x = rng.integers(0, data.shape[0])
                seen.add(x)
                centroids[i] = data[x]

            centroids, assignments = kmeans(data, centroids, max_iterations)

            # Store potential cluster assingment and its inertia(how good it is)
            results.append((centroids, assignments, calc_inertia(data, centroids, assignments))) 
        # best is with loewst inertia
        best = min(results, key=lambda t: t[2])
        plot_kmeans(data, best[0], best[1], f"kmeans_k{k}.png", output_dir)
        


    


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-iterations", type=int, default=100)
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).resolve().parent / "results" / "kmeans")
    args = parser.parse_args()
    run_experiments(args.output_dir, args.seed, args.max_iterations)


if __name__ == "__main__":
    main()
