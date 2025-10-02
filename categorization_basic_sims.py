import numpy as np
import pandas as pd
from scipy.spatial.distance import squareform, cdist, pdist
from scipy import stats
from itertools import combinations
import sklearn.metrics as metrics
from treelib import Tree
import os

class SimCluster:
    def __init__(self, simMat, imgInfo):
        self.simMat = simMat
        self.imgInfo = imgInfo

        # Figure out level map from imgInfo
        self.levelMap = {
            "super": list(imgInfo["super"].dropna().unique()),
            "basic": list(imgInfo["basic"].dropna().unique()),
            "sub": list(imgInfo["sub"].dropna().unique()),
        }

        # Figure out sets
        self.sets = list(imgInfo["set"].unique())

    def calculate_index(
        self, imgSet=None, level=None, category=None, within_level=False
    ):
        # Both level and category cannot be set together
        if level is not None and category is not None:
            raise ValueError("Both level and category cannot be set together")

        if imgSet is not None:
            # Filter imgInfo by sets
            imgInfo = self.imgInfo[self.imgInfo["set"] == imgSet]
        else:
            imgInfo = self.imgInfo

        # Handle average level indices first
        if level is not None:
            # Get categories
            categories = self.levelMap[level]

            # Preallocate array for cluster indices
            clusters = np.zeros(len(categories), dtype=np.float32)
            # Loop through categories
            for k, cat in enumerate(categories):
                loc = imgInfo[level] == cat
                withinIdxs = imgInfo.loc[loc, "name"].index

                if within_level and level != "super":
                    hier = list(self.levelMap.keys())
                    higherLevel = hier[hier.index(level) - 1]

                    # Get the higher level category
                    higherCat = imgInfo.loc[withinIdxs, higherLevel].unique()[0]

                    loc = (imgInfo[level] != cat) & (imgInfo[higherLevel] == higherCat)
                    betweenIdxs = imgInfo.loc[loc, "name"].index
                else:
                    loc = imgInfo[level] != cat
                    betweenIdxs = imgInfo.loc[loc, "name"].index

                withinSum = 0
                withinCount = 0
                for i, j in combinations(withinIdxs, 2):
                    withinSum += self.simMat[i, j]
                    withinCount += 1

                betweenSum = 0
                betweenCount = 0
                for i in withinIdxs:
                    for j in betweenIdxs:
                        betweenSum += self.simMat[i, j]
                        betweenCount += 1

                clusters[k] = (withinSum / withinCount) - (betweenSum / betweenCount)

            return np.mean(clusters)
        elif category is not None:
            # Find the level of the category
            for level, categories in self.levelMap.items():
                if category in categories:
                    break

            loc = imgInfo[level] == category
            withinIdxs = imgInfo.loc[loc, "name"].index

            if within_level and level != "super":
                hier = list(self.levelMap.keys())
                higherLevel = hier[hier.index(level) - 1]

                # Get the higher level category
                higherCat = imgInfo.loc[withinIdxs, higherLevel].unique()[0]

                loc = (imgInfo[level] != category) & (imgInfo[higherLevel] == higherCat)
                betweenIdxs = imgInfo.loc[loc, "name"].index
            else:
                loc = imgInfo[level] != category
                betweenIdxs = imgInfo.loc[loc, "name"].index

            withinSum = 0
            withinCount = 0
            for i, j in combinations(withinIdxs, 2):
                withinSum += self.simMat[i, j]
                withinCount += 1

            betweenSum = 0
            betweenCount = 0
            for i in withinIdxs:
                for j in betweenIdxs:
                    betweenSum += self.simMat[i, j]
                    betweenCount += 1

            return (withinSum / withinCount) - (betweenSum / betweenCount)
        else:
            raise ValueError("Either level or category must be set")

    def calculate_all(self, within_level=False):
        for imgSet in self.sets:
            for level in self.levelMap.keys():
                for category in self.levelMap[level]:
                    val = self.calculate_index(
                        imgSet=imgSet, category=category, within_level=within_level
                    )
                    print(f"{imgSet}-{level}-{category}: {val}")
            print("--")

def get_nodes_at_level(tree, level):
    """Return a list of nodes at a given level of the tree"""
    return [
        i.identifier for i in tree.all_nodes_itr() if tree.level(i.identifier) == level
    ]


def get_leaves_from_node(tree, node):
    """Return the indices of the items (leaves) from a given node"""
    leafList = [leaf.data["indices"] for leaf in tree.leaves(node)]
    return np.concatenate(leafList)

def external_evaluate_over_levels(tree, labels, metric, verbose=False):
    levels = range(1, labels.shape[1] + 1)
    nLeaves = len(tree.get_node(0).data["indices"])

    scores = np.zeros(len(levels))
    for i, level in enumerate(levels):
        levelLabels = labels[:, level - 1]
        nodes = get_nodes_at_level(tree, level)
        levelPred = np.repeat(-1, nLeaves)
        for j, node in enumerate(nodes):
            leaves = get_leaves_from_node(tree, node)

            levelPred[leaves] = j

        # Calculate external metric
        score = metric(levelLabels, levelPred)
        scores[i] = score

        if verbose:
            print(f"Level {level}: {score}")

    return scores


def internal_evaluate_over_levels(tree, reps, metric, level=None, verbose=False):
    if level is None:
        maxLevel = max([tree.level(i.identifier) for i in tree.all_nodes_itr()]) + 1
    else:
        maxLevel = level + 1
    levels = range(1, maxLevel)
    nLeaves = len(tree.get_node(0).data["indices"])

    scores = np.zeros(len(levels))
    for i, level in enumerate(levels):
        nodes = get_nodes_at_level(tree, level)
        levelPred = np.repeat(-1, nLeaves)
        for j, node in enumerate(nodes):
            leaves = get_leaves_from_node(tree, node)

            levelPred[leaves] = j

        # Calculate internal metric
        score = metric(reps, levelPred)
        scores[i] = score

        if verbose:
            print(f"Level {level}: {score}")

    return scores


def calc_cue_validity(exemplars, labels, binary=True, verbose=False):
    categories = np.unique(labels)

    cueValidities = {}
    for category in categories:
        cueValidity = 0
        for k in range(exemplars.shape[1]):
            if binary:
                hasFeature = exemplars[:, k] > 0

                # Check how many images with this feature are in this category
                nImages = np.sum(hasFeature[labels == category])

                cueValidity += nImages / exemplars.shape[0]
            else:
                # Binarize label
                catLabels = np.float32(labels == category)

                # Get features
                featureStrength = exemplars[:, k]

                # Calculate point biserial correlation with fisher Z transform
                cueValidity += np.abs(
                    np.arctanh(np.corrcoef(featureStrength, catLabels)[0, 1])
                )

        cueValidities[category] = cueValidity / exemplars.shape[1]

        if binary:
            # Z to r
            cueValidities[category] = np.tanh(cueValidities[category])

        if verbose:
            print(
                "Category: ",
                category,
                "Cue validity: ",
                np.abs(cueValidity) / exemplars.shape[1],
            )

    return cueValidities


def calc_category_validity(exemplars, labels, binary=True, verbose=False):
    categories = np.unique(labels)

    categoryValidities = {}
    for category in categories:
        categoryImgs = exemplars[category == labels, :]
        category_validity = 0
        for k in range(exemplars.shape[1]):
            if binary:
                # Check how many images has this feature
                hasFeature = np.sum(categoryImgs[:, k] > 0)

                # Add to category_validity
                category_validity += hasFeature / categoryImgs.shape[0]
            else:
                # Average the feature absolute strength
                category_validity += np.mean(np.abs(categoryImgs[:, k]))

        # Save
        categoryValidities[category] = category_validity / exemplars.shape[1]
        if verbose:
            print(
                "Category: ",
                category,
                " Validity: ",
                category_validity / exemplars.shape[1],
            )

    return categoryValidities


def calc_collocation(exemplars, labels, binary=True, verbose=False):
    categories = np.unique(labels)

    collocations = {}
    for category in categories:
        categoryImgs = exemplars[category == labels, :]
        category_validity = 0
        cueValidity = 0
        for k in range(exemplars.shape[1]):
            if binary:
                # Cue validity
                hasFeature = exemplars[:, k] > 0

                # Check how many images with this feature are in this category
                nImages = np.sum(hasFeature[labels == category])

                cueValidity += nImages / exemplars.shape[0]

                # Category validity
                hasFeature = np.sum(categoryImgs[:, k] > 0)

                # Add to category_validity
                category_validity += hasFeature / categoryImgs.shape[0]
            else:
                # Calculate cue validity
                catLabels = np.float32(labels == category)

                # Get features
                features = exemplars[:, k]

                # Calculate correlation
                cueValidity += np.abs(
                    np.arctanh(np.corrcoef(catLabels, features)[0, 1])
                )

                # Calculate category validity
                category_validity += np.mean(np.abs(categoryImgs[:, k]))

        # Divide by number of features
        category_validity /= exemplars.shape[1]
        cueValidity /= exemplars.shape[1]

        collocations[category] = category_validity * cueValidity

        if verbose:
            print(
                "Category: ",
                category,
                " Collocation: ",
                category_validity * cueValidity,
            )

    return collocations


def calc_category_utility(exemplars, labels, binary=True, verbose=False):
    categories = np.unique(labels)

    category_utilities = {}
    for category in categories:
        # Calculate the frequency of this category amongst all labels
        category_frequency = np.sum(labels == category) / labels.shape[0]

        # Loop through features
        category_utility = 0
        for k in range(exemplars.shape[1]):
            categoryImgs = exemplars[category == labels, :]
            if binary:
                # Calculate the frequency that an image has this feature
                feature_frequency = np.sum(exemplars[:, k] > 0) / exemplars.shape[0]

                # Calculate category validity
                hasFeature = np.sum(categoryImgs[:, k] > 0)
                category_validity = hasFeature / categoryImgs.shape[0]

                # Add to category validity
                category_utility += (category_validity**2) - (feature_frequency**2)
            else:
                # Calculate average feature strength (regardless of category)
                feature_strength = np.mean(np.abs(exemplars[:, k]))

                # Calculate category validity
                category_validity = np.mean(np.abs(categoryImgs[:, k]))

                # Category utility
                category_utility += category_validity - feature_strength

        # Multiply by category frequency
        category_utilities[category] = category_utility * category_frequency

        if verbose:
            print(
                "Category: ",
                category,
                " Utility: ",
                category_utility * category_frequency,
            )

    return category_utilities


def print_cluster_stats(tree, hierLabels, exemplars):
    """
    HierLabels must be formatted nxk where k are the levels of the hierarchy
    """
    # Adjusted rand score, 1 is perfect, pair counting method
    print("Adjusted Rand score:")
    _ = external_evaluate_over_levels(
        tree, hierLabels, metrics.adjusted_rand_score, verbose=True
    )

    # Mutual information, 1 is perfect, agreement between two partitions
    print("Adjusted Mutual information:")
    _ = external_evaluate_over_levels(
        tree, hierLabels, metrics.adjusted_mutual_info_score, verbose=True
    )

    # V-measure, 1 is perfect, weighted harmonic mean of homogeneity (cluster only includes one class) and completeness (all members in one class)
    print("V-measure:")
    _ = external_evaluate_over_levels(
        tree, hierLabels, metrics.v_measure_score, verbose=True
    )

    # Fowlkes-Mallows, 1 is perfect, geometric mean between precision (TP/TP+FP) and recall (TP/FP+FN)
    print("Fowlkes-Mallows:")
    _ = external_evaluate_over_levels(
        tree, hierLabels, metrics.fowlkes_mallows_score, verbose=True
    )

    # Davies-Bouldin, 0 is best partitioning, signifies average similarity between clusters
    print("Davies-Bouldin:")
    _ = internal_evaluate_over_levels(
        tree,
        exemplars,
        metrics.davies_bouldin_score,
        level=3,
        verbose=True,
    )

    # Silhouette score, 0 is overlapping clusters, +1 is perfect clustering, -1 is wrong clustering
    print("Silhouette score:")
    _ = internal_evaluate_over_levels(
        tree,
        exemplars,
        metrics.silhouette_score,
        level=3,
        verbose=True,
    )

    # Calinski_harabasz, higher is denser well-separated clusters
    print("Calinski-Harabasz:")
    _ = internal_evaluate_over_levels(
        tree,
        exemplars,
        metrics.calinski_harabasz_score,
        level=3,
        verbose=True,
    )

    return None


def print_category_metrics(exemplars, labels, simMat, imgInfo, binary=True):
    """
    Labels must be formatted nx3 where the second dimension is super, basic, and sub
    """
    superLabels = labels[:, 0]
    basicLabels = labels[:, 1]
    subLabels = labels[:, 2]

    print("Category cue validity = Sum(P(C|fk)) / n")
    print("Superordinate: ")
    calc_cue_validity(exemplars, superLabels, binary=binary, verbose=True)

    print("Basic: ")
    calc_cue_validity(exemplars, basicLabels, binary=binary, verbose=True)

    print("Subordinate: ")
    _ = calc_cue_validity(exemplars, subLabels, binary=binary, verbose=True)
    print()

    print("Category validity = Sum(P(fk|C)) / n")
    print("Superordinate: ")
    calc_category_validity(exemplars, superLabels, binary=binary, verbose=True)

    print("Basic: ")
    calc_category_validity(exemplars, basicLabels, binary=binary, verbose=True)

    print("Subordinate: ")
    _ = calc_category_validity(exemplars, subLabels, binary=binary, verbose=True)
    print()

    print("Collocation (cue validity * category validity)")
    print("Superordinate: ")
    calc_collocation(exemplars, superLabels, binary=binary, verbose=True)

    print("Basic: ")
    calc_collocation(exemplars, basicLabels, binary=binary, verbose=True)

    print("Subordinate: ")
    _ = calc_collocation(exemplars, subLabels, binary=binary, verbose=True)
    print()

    print("Category utility P(C) * Sum(P(Fk|C) ** 2 - P(Fk)**2)")
    print("Superordinate: ")
    calc_category_utility(exemplars, superLabels, binary=binary, verbose=True)

    print("Basic: ")
    calc_category_utility(exemplars, basicLabels, binary=binary, verbose=True)

    print("Subordinate: ")
    _ = calc_category_utility(exemplars, subLabels, binary=binary, verbose=True)
    print()

    print("Cluster index (mean within sim - mean betwen sim)")
    simCluster = SimCluster(simMat=simMat, imgInfo=imgInfo)

    print("Superordinate: ")
    print(simCluster.calculate_index(level="super"))

    print("Basic: ")
    print(simCluster.calculate_index(level="basic"))

    print("Subordinate: ")
    print(simCluster.calculate_index(level="sub"))

def exemplar_maker(n, center, radius=1, radius_density="uniform", relu=False):
    nDims = len(center)

    # Generate random numbers as needed
    coords = np.random.normal(loc=0, scale=1, size=(n, nDims))
    uniforms = np.random.uniform(low=0, high=1, size=n)

    if radius_density == "power":
        radii = (uniforms ** (1 / nDims)) * radius
    elif radius_density == "normal":
        radii = np.abs(np.random.normal(loc=0, scale=1, size=n)) * radius
    elif radius_density == "uniform":
        radii = uniforms * radius
    elif radius_density == "lognormal":
        radii = np.random.lognormal(mean=0, sigma=1 / 3, size=n) * radius
    else:
        raise ValueError("Density type not recognized")

    coords = coords.T / np.linalg.norm(
        coords, axis=1
    )  # Uniformly distributed directions
    coords = coords * radii  # Change radii
    coords = coords.T + center

    # If relu, apply relu
    if relu:
        coords[coords < 0] = 0

    return coords

def make_categories(
    *,
    cat_rad,
    radius_density="power",
    relu=False,
    super_rad,
    basic_rad,
    sub_rad,
    nFeatures,
    nImages,
):
    def _centroids_maker(center, r):
        """
        Create two centroids on a surface of a hypersphere with radius r. The
        first centroid is randomly selected from the surface of a n-sphere
        """
        nFeatures = center.shape[0]

        coords = stats.multivariate_normal.rvs(mean=np.zeros((nFeatures,)), cov=1)

        # Change coordinates to unit length
        coords = coords / np.linalg.norm(coords)

        # Multiply coords by radius of sphere
        coords = coords * r

        # Return coordinates plus and minus center
        return center + coords, center - coords

    # Make superordinate centroids
    superCentroids = _centroids_maker(
        center=np.zeros((nFeatures,), dtype=np.float32), r=super_rad
    )

    # Make basic centroids
    basicCentroids = np.zeros((4, nFeatures), dtype=np.float32)
    for i, center in enumerate(superCentroids):
        basicCentroids[(i * 2) : (i * 2 + 2)] = _centroids_maker(
            center=center, r=basic_rad
        )

    # Make subordinate centroids
    subCentroids = np.zeros((8, nFeatures), dtype=np.float32)
    for i, center in enumerate(basicCentroids):
        subCentroids[(i * 2) : (i * 2 + 2)] = _centroids_maker(center=center, r=sub_rad)

    # Generate exemplars
    subExemplars = np.zeros((nImages * 8, nFeatures), dtype=np.float32)
    subLabels = np.zeros((nImages * 8,), dtype=np.int32)
    for i, center in enumerate(subCentroids):
        subExemplars[(i * nImages) : (i * nImages + nImages)] = exemplar_maker(
            nImages,
            center=center,
            radius=cat_rad,
            radius_density=radius_density,
            relu=relu,
        )
        subLabels[(i * nImages) : (i * nImages + nImages)] = i

    return subExemplars, subCentroids, subLabels

class diana:
    def __init__(self, data, metric, max_clusters=None, verbose=False):
        self.data = data
        self.metric = metric
        indices = np.arange(data.shape[0])
        self.tree = Tree()
        self.verbose = verbose

        self.tree.create_node(
            "root",
            0,
            data={
                "indices": indices,
            },
        )

        if max_clusters is None:
            max_clusters = data.shape[0]

        while len(self.tree.leaves()) < max_clusters:
            if self.verbose:
                print(
                    f"We have {len(self.tree.leaves())} clusters, running diana step..."
                )
            # Pick cluster with largest diameter
            nid = self.pick_cluster().identifier

            # Split cluster
            self.split_cluster(nid)

    def _mean_diss(self, simMatrix):
        return np.sum(simMatrix, axis=0) / (simMatrix.shape[0] - 1)

    def split_cluster(self, nid):
        node = self.tree.get_node(nid)

        oldCluster = np.copy(node.data["indices"])
        clusterSim = squareform(pdist(self.data[oldCluster,], metric=self.metric))

        # Find the item that is most dissimilar to the rest of the cluster
        mostDissIdx = np.argmax(self._mean_diss(clusterSim))
        newCluster = oldCluster[mostDissIdx]

        # Remove most dissimilar index from old cluster
        oldCluster = np.delete(oldCluster, mostDissIdx)

        while len(oldCluster) > 1:
            # Compute dissimilarity of old cluster
            oldDiss = squareform(pdist(self.data[oldCluster,], metric=self.metric))
            oldDiss = self._mean_diss(oldDiss)

            # Now compute similarity of each item in the old cluster with the new cluster
            oldClusterData = self.data[oldCluster, :]
            newClusterData = self.data[newCluster, :]

            # if new cluster data is 1D, reshape to 2D
            if len(newClusterData.shape) == 1:
                newClusterData = newClusterData.reshape(1, -1)

            newDiss = (
                np.sum(
                    cdist(oldClusterData, newClusterData, metric=self.metric),
                    axis=1,
                )
                / newClusterData.shape[0]
            )

            # Find new item to remove from old cluster
            dissDiff = oldDiss - newDiss
            mostDissIdx = np.argmax(dissDiff)

            # Check if most dissimilar item is more dissimilar than the new cluster
            if dissDiff[mostDissIdx] < 0:
                break

            # Update clusters
            newCluster = np.append(newCluster, oldCluster[mostDissIdx])
            oldCluster = np.delete(oldCluster, mostDissIdx)

        # Figure out level
        level = self.tree.level(nid) + 1

        # Figure out how many nodes are at this level
        nodesAtLevel = len(
            [
                node
                for node in self.tree.all_nodes()
                if self.tree.level(node.identifier) == level
            ]
        )

        # Figure out the highest nid
        highestNid = np.max([node.identifier for node in self.tree.all_nodes()])

        self.tree.create_node(
            f"level{level}.{nodesAtLevel}",
            highestNid + 1,
            parent=nid,
            data={
                "indices": oldCluster,
            },
        )

        # If new cluster is only 1 element, make it an array
        if not isinstance(newCluster, np.ndarray):
            newCluster = np.array([newCluster])

        self.tree.create_node(
            f"level{level}.{nodesAtLevel + 1}",
            highestNid + 2,
            parent=nid,
            data={
                "indices": newCluster,
            },
        )

        if self.verbose:
            print(
                f"Split cluster {nid} into {highestNid + 1} and {highestNid + 2} at level {level}"
            )
            print(f"Cluster {highestNid + 1} has {len(oldCluster)} objects")
            print(f"Cluster {highestNid + 2} has {len(newCluster)} objects")

    def pick_cluster(self):
        # Get every leaf
        leaves = self.tree.leaves()

        # Calculate diameter of each leaf
        diameters = np.zeros(len(leaves))
        for i, leaf in enumerate(leaves):
            leafData = self.data[leaf.data["indices"], :]
            if len(leafData) == 1:
                diameters[i] = 0
            else:
                diameters[i] = np.max(pdist(leafData, metric=self.metric))

        # Pick the leaf with the largest diameter
        return leaves[np.argmax(diameters)]

    def prune_tree(self, level):
        level += 1
        # Loop through all nodes and delete nodes just after the target level
        for node in self.tree.all_nodes():
            if (
                self.tree.get_node(node.identifier) is not None
                and self.tree.level(node.identifier) == level
            ):
                self.tree.remove_node(node.identifier)

    def linkage_matrix(self, calc_dist=False):
        # Copy tree
        tree = Tree(self.tree.subtree(self.tree.root), deep=True)

        nData = self.data.shape[0]
        # Start building linkage matrix
        linkage = np.zeros((nData - 1, 4))
        rowCount = 0

        # Loop through leaves
        for leaf in tree.leaves():
            # Each leaf is its own cluster, so stick together every object into a bigger and bigger cluster
            cluster = leaf.data["indices"]

            # If the cluster is only one object, just give it a nodeID of itself
            if len(cluster) == 1:
                leaf.data["linkID"] = cluster[0]
                continue

            # Calculate the average distance between objects in the cluster
            if calc_dist:
                clusterReps = self.data[cluster, :]
                clusterDist = np.mean(pdist(clusterReps, metric=self.metric))
            else:
                clusterDist = 0.2

            # Stick the first two items together into a new cluster
            linkage[rowCount, 0] = cluster[0]
            linkage[rowCount, 1] = cluster[1]
            linkage[rowCount, 2] = clusterDist
            linkage[rowCount, 3] = 2
            rowCount += 1

            # Loop through the remaining items and stick it to this cluster
            for i in range(2, len(cluster)):
                linkID = rowCount + nData - 1
                linkage[rowCount, 0] = cluster[i]
                linkage[rowCount, 1] = linkID
                linkage[rowCount, 2] = clusterDist
                linkage[rowCount, 3] = linkage[rowCount - 1, 3] + 1
                rowCount += 1

            # Remember the linkID for this leaf cluster
            leaf.data["linkID"] = rowCount + nData - 1

        # Now loop through the tree and build the rest of the linkage matrix
        for i in range(len(tree.nodes) - 1, -1, -1):
            if i == 0:
                continue

            # if tree.get_node(i) is None:
            #     continue

            # Get the node's parent
            ancestor = tree.get_node(tree.ancestor(i))

            # Only work on this node if the parent doesn't have a linkID yet
            if not "linkID" in ancestor.data:
                # Get the node
                node1 = tree.get_node(i)

                # Get the node's sibling
                node2 = tree.siblings(i)[0]

                # # If sibling doesn't have link ID, skip this for now
                # if not "linkID" in node2.data:
                #     continue

                # Calculate the mean distance bewteen the objects in each node
                if calc_dist:
                    node1Reps = self.data[node1.data["indices"]]
                    node2Reps = self.data[node2.data["indices"]]
                    nodeDist = np.mean(cdist(node1Reps, node2Reps, self.metric))
                else:
                    nodeDist = tree.depth() - tree.level(i) + 1

                # Add the new entry to linkage
                linkID = rowCount + nData
                linkage[rowCount, 0] = node1.data["linkID"]
                linkage[rowCount, 1] = node2.data["linkID"]
                linkage[rowCount, 2] = nodeDist
                linkage[rowCount, 3] = len(ancestor.data["indices"])
                rowCount += 1

                # Save the linkID to the ancestor
                ancestor.data["linkID"] = linkID

        return linkage

class EBRW:
    def __init__(
        self,
        memory_reps: np.ndarray,
        memory_categories: np.ndarray,
        rng: np.random.Generator,
        memory_strengths: np.ndarray = None,
        memory_strength_multiplier: float = 1.0,
        p: float = 2.0,
        c: float = 1.0,
        b: float = 0,
        A: float = 10,
        B: float = 10,
        alpha: float = 1,
    ):
        """
        Return an instance of the EBRW model starting with the given memory
        represntations and their categories (index labels).
        """
        # Memory
        self.memory_reps = memory_reps
        self.memory_categories = memory_categories
        self.memory_strengths = (
            memory_strengths
            if memory_strengths is not None
            else np.ones(len(memory_categories)) / len(memory_categories)
        )
        self.memory_strengths *= memory_strength_multiplier
        self.categories = np.unique(memory_categories)

        if len(self.categories) > 2:
            raise ValueError("We only supports binary categorization")

        self.rng = rng

        # Model parameters
        self.p = p  # Distance metric
        self.c = c  # Sensitivity
        self.b = b  # Criterion/background
        self.A = A  # Category 0 threshold
        self.B = B  # Category 1 threshold
        self.alpha = alpha  # Step time constant

    def _sim(self, probes: np.ndarray, category: int, metric="minkowski", **kwargs):
        """
        Calculates the similarity between the probe items and the
        represenations in category. Calculates distance using cdist with
        defaults for EBRW. Chiefly, default Minkowski distance with p=2 and
        w=1/n_features. The metric and kwargs can be changed to modify this. The
        distance is then used to calculate similarity alongside the c parameter.
        """
        if metric == "minkowski":
            if "p" not in kwargs.keys():
                kwargs["p"] = self.p

            if "w" not in kwargs.keys():
                kwargs["w"] = np.ones(probes.shape[1]) / probes.shape[1]

        # Get memory_reps that match the category
        category_reps = self.memory_reps[self.memory_categories == category]

        if len(category_reps) == 0:
            return np.zeros((probes.shape[0],))

        # Calculate distances
        dists = cdist(probes, category_reps, metric=metric, **kwargs)

        # Calculate similarity
        return (
            np.exp(-self.c * dists)
            * self.memory_strengths[self.memory_categories == category]
        )

    def _sum_sims(self, probes, **kwargs):
        """
        Return the sum similarities given the probes.
        """
        # Calculate sum similarities
        sumA = np.sum(self._sim(probes, self.categories[0], **kwargs), axis=1)

        if len(self.categories) == 1:
            return sumA, 0
        else:
            sumB = np.sum(self._sim(probes, self.categories[1], **kwargs), axis=1)

            return sumA, sumB

    def _prob_step(self, sumA, sumB):
        """
        Return the probability of stepping towards the category from the sum
        similarity.
        """
        p = (sumA + self.b) / (sumA + sumB + (self.b * 2))

        return p, 1 - p

    def categorize(self, probes, categories, **kwargs):
        """
        Return the responses and RT for each probe targetting each category.
        """
        # First calculate sum similarities
        sumA, sumB = self._sum_sims(probes, **kwargs)

        # Calculate step probabilities
        p, q = self._prob_step(sumA, sumB)

        # Calculate probability of category choices
        top = 1 - ((q / p) ** self.B)
        bot = 1 - ((q / p) ** (self.A + self.B))
        pA = top / bot

        top = ((q / p) ** self.B) - ((q / p) ** (self.A + self.B))
        bot = 1 - ((q / p) ** (self.A + self.B))
        pB = top / bot

        # Calculate expected number of steps for each type of response
        # Calculate steps A
        top = ((p / q) ** (self.A + self.B)) + 1
        bot = ((p / q) ** (self.A + self.B)) - 1
        theta1A = top / bot

        top = ((p / q) ** self.B) + 1
        bot = ((p / q) ** self.B) - 1
        theta2A = top / bot

        top = (theta1A * (self.A + self.B)) - (theta2A * self.B)
        bot = p - q
        stepsA = top / bot

        # Calculate steps B
        top = ((p / q) ** -(self.A + self.B)) + 1
        bot = ((p / q) ** -(self.A + self.B)) - 1
        theta1B = top / bot

        top = ((p / q) ** -self.A) + 1
        bot = ((p / q) ** -self.A) - 1
        theta2B = top / bot

        top = (theta1B * (self.A + self.B)) - (theta2B * self.A)
        bot = q - p
        stepsB = top / bot

        # Stick the steps together
        steps = np.stack([stepsA, stepsB], axis=1)

        # Calculate step time
        stepTime = (self.alpha + 1) / (sumA + sumB)

        # Calculate decisions
        decisions = np.zeros((probes.shape[0],), dtype=np.int32)
        rts = np.zeros((probes.shape[0],))
        for i in range(probes.shape[0]):
            if categories[i] == self.categories[0]:
                decision = self.rng.choice([0, 1], p=[pA[i], 1 - pA[i]]).astype(
                    np.int32
                )
                decisions[i] = decision
                rts[i] = steps[i, decision] * stepTime[i]
            else:
                decision = self.rng.choice([0, 1], p=[pB[i], 1 - pB[i]]).astype(
                    np.int32
                )
                decisions[i] = decision
                rts[i] = steps[i, decision] * stepTime[i]

        return decisions, rts

def simulate_task(
    imgInfo: pd.DataFrame,
    *,
    rng: np.random.Generator,
    exemplars: np.ndarray = None,
    centroids: np.ndarray = None,
    balanced: bool = True,
    nreps: int = 100,
    foils: bool = False,
    exemplar_kwargs: dict = {},
    ebrw_kwargs: dict = {},
) -> pd.DataFrame:
    """
    Return results from simulating the category verification task.

    Parameters
    ----------
    imgInfo : pd.DataFrame
        A dataframe with at least the columns "super", "basic", "sub", "set". If
        "image" is present, it will be used to label the images in the task,
        otherwise, representations are assumed to be simulated.
    rng: np.random.Generator
        A numpy random number generator, passed along to any functions needing
        the RNG.
    exemplars : np.ndarray, optional
        A 2D array of shape (nImages, nFeatures) with the exemplars.
    centroids : np.ndarray, optional
        A 2D array of shape (nCategories, nFeatures) with the centroids.
    balanced : bool, optional
        Whether to balance the memory exemplars across levels.
    nreps : int, optional
        The number of representations for the target (and foil) category in
        memory at each level.
    foils : bool, optional
        Whether to include foils in the task.
    exemplar_kwargs : dict, optional
        Keyword arguments to pass to the exemplar_maker function.
    ebrw_kwargs : dict, optional
        Keyword arguments to pass to the EBRW model.

    Returns
    -------
    pd.DataFrame
        A dataframe with columns "image", "category", "level", "response", "RT",
        containing the simulated task results.
    """

    def _bootstrapReps(reps, repIdxs, subs):
        # Bootstraps representations based on subordinates
        nSim = int(nreps / len(subs))
        if nSim % 1 != 0:
            raise ValueError(
                "nreps must be divisible by the number of subordinates at every level"
            )
        else:
            nSim = int(nSim)

        bootInfo = imgInfo.loc[repIdxs].reset_index(drop=True)

        bootReps = []
        for sub in subs:
            # Get representations for the subordinates
            subIdxs = bootInfo[bootInfo["sub"] == sub].index
            subReps = reps[subIdxs]

            # Bootstrap the representations
            bootReps += [rng.choice(subReps, size=nSim, replace=True)]

        return np.concatenate(bootReps, axis=0)

    if (exemplars is not None and centroids is not None) or (
        exemplars is None and centroids is None
    ):
        raise ValueError("Either exemplars or centroids must be provided")
    else:  # Temporary measure to make highlighting work
        _ = 1 + 1

    if centroids is not None:
        # Check if imgInfo has the right subordinate category labels
        subs = imgInfo["sub"].unique()
        if not np.all(np.array(range(len(centroids))) == subs):
            raise ValueError("Centroids do not match the subordinate category labels")

    # Figure out how many exemplars are in memory
    nMemory = len(imgInfo.loc[imgInfo["set"] == "train"])

    # Setup dataframe
    performance = pd.DataFrame(columns=["image", "category", "level", "response", "RT"])

    # Loop through levels
    levels = ["super", "basic", "sub"]
    for i, level in enumerate(levels):
        # Get categories at this level
        cats = imgInfo[level].unique()

        # Loop through categories
        for category in cats:
            if exemplars is not None:
                # Find the test exemplars for this category
                testIdxs = imgInfo.loc[
                    (imgInfo[level] == category) & (imgInfo["set"] == "test")
                ].index
                testReps = exemplars[testIdxs]

                # Find the memory exemplars for this category
                memoryIdxs = imgInfo.loc[
                    (imgInfo[level] == category) & (imgInfo["set"] == "train")
                ].index
                memoryReps = exemplars[memoryIdxs]
                memoryLabels = np.zeros(len(memoryReps))

                if foils:  # EBRW task with foils
                    foilIdx = imgInfo.loc[
                        (imgInfo[level] != category) & (imgInfo["set"] == "train")
                    ].index

                    # Only keep the foils with the same parent category
                    if level != "super":  # Keep all for super
                        parentLevel = levels[levels.index(level) - 1]
                        parentCat = imgInfo.iloc[memoryIdxs[0]][parentLevel]
                        validIdxs = imgInfo.loc[imgInfo[parentLevel] == parentCat].index

                        # Only keep foilIDxs in validIdxs
                        foilIdx = [x for x in foilIdx if x in validIdxs]

                    foilReps = exemplars[foilIdx]
                    foilLabels = np.ones(len(foilReps))

                if balanced:
                    # Split the memory representations by subordinate category
                    memorySubs = imgInfo.loc[memoryIdxs, "sub"].unique()

                    memoryReps = _bootstrapReps(memoryReps, memoryIdxs, memorySubs)
                    memoryLabels = np.zeros(len(memoryReps))

                    if foils:
                        # Split the foil representations by subordinate category
                        foilSubs = imgInfo.loc[foilIdx, "sub"].unique()

                        foilReps = _bootstrapReps(foilReps, foilIdx, foilSubs)
                        foilLabels = np.ones(len(foilReps))

                if foils:
                    memoryReps = np.concatenate([memoryReps, foilReps], axis=0)
                    memoryLabels = np.concatenate([memoryLabels, foilLabels], axis=0)
            else:  # Using centroids to create balanced levels
                if balanced:
                    Warning("balanced argument has no effect on centroids.")

                # Find the subordinates for this category
                subs = imgInfo.loc[imgInfo[level] == category, "sub"].unique()
                nSims = int(nreps / len(subs))

                if nSims % 1 != 0:
                    raise ValueError(
                        "nreps must be divisible by the number of subordinates at every level"
                    )

                # Simulate some reps from each centroid
                testReps = np.concatenate(
                    [
                        exemplar_maker(nSims, centroids[sub], **exemplar_kwargs)
                        for sub in subs
                    ],
                    axis=0,
                )
                memoryReps = np.concatenate(
                    [
                        exemplar_maker(nSims, centroids[sub], **exemplar_kwargs)
                        for sub in subs
                    ],
                    axis=0,
                )
                memoryLabels = np.zeros(len(memoryReps))

                # Make foils if necessary
                if foils:
                    # Find the subordinates for the foils
                    foilSubs = imgInfo.loc[imgInfo[level] != category, "sub"].unique()

                    # Only keep the foils with the same parent category
                    if level != "super":
                        parentLevel = levels[levels.index(level) - 1]
                        parentCat = imgInfo.loc[imgInfo["sub"] == subs[0]][
                            parentLevel
                        ].iloc[0]
                        foilSubs = [
                            sub
                            for sub in foilSubs
                            if imgInfo[parentLevel].loc[imgInfo["sub"] == sub].iloc[0]
                            == parentCat
                        ]

                    # Make representations
                    foilReps = np.concatenate(
                        [
                            exemplar_maker(nSims, centroids[sub], **exemplar_kwargs)
                            for sub in foilSubs
                        ],
                        axis=0,
                    )
                    foilLabels = np.ones(len(foilReps))

                    # Combine foils with memory
                    memoryReps = np.concatenate([memoryReps, foilReps], axis=0)
                    memoryLabels = np.concatenate([memoryLabels, foilLabels], axis=0)

            # Handle the possibility of lists in kwargs
            _ebrwKwargs = {}
            for key, value in ebrw_kwargs.items():
                if isinstance(value, list):
                    _ebrwKwargs[key] = value[i]
                else:
                    _ebrwKwargs[key] = value
            ebrw = EBRW(
                memory_reps=memoryReps,
                memory_categories=memoryLabels,
                memory_strengths=np.ones(len(memoryReps)) * (1 / nMemory),
                rng=rng,
                **_ebrwKwargs,
            )

            decisions, rts = ebrw.categorize(
                probes=testReps, categories=np.zeros(len(testReps))
            )

            # Check if images have names
            if "image" in imgInfo.columns:
                imageNames = imgInfo.loc[testIdxs, "image"]
            else:
                imageNames = "simulated"

            tmp = pd.DataFrame(
                {
                    "image": imageNames,
                    "category": [category] * len(testReps),
                    "level": [level] * len(testReps),
                    "response": ["yes" if x == 0 else "no" for x in decisions],
                    "RT": rts,
                }
            )
            performance = pd.concat([performance, tmp], axis=0)

    return performance

def create_simulated_image_info():
    """
    Create simulated image info DataFrame for categorization experiments.
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with columns 'super', 'basic', 'sub', 'set' containing
        hierarchical category labels and train/test split information.
    """
    # Make subordinate labels, a new index every 100 images
    subLabels = np.repeat(np.arange(8), 100)

    # Designate half of the images as test and train, alternating every 50
    trainTest = np.concatenate([np.repeat(["train", "test"], 50) for i in range(8)])

    # Make basic and super labels
    # Every 2 subordinate are the same basic level category
    basicLabels = np.repeat(np.arange(4), 200)

    # Every 2 basic level categories are the same super level category
    superLabels = np.repeat(np.arange(2), 400)

    # Stick together labels
    simInfo = pd.DataFrame(
        {
            "super": superLabels,
            "basic": basicLabels,
            "sub": subLabels,
            "set": trainTest,
        }
    )
    
    return simInfo

def run_ebrw_simulation(ebrw_kwargs, simReps, file_name, seeds=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]):
    """
    Run EBRW simulation with given parameters and return performance summaries.
    
    Parameters:
    -----------
    ebrw_kwargs : dict
        Parameters for EBRW model (e.g., {"c": 1, "b": 0.25})
    simReps : np.ndarray
        Exemplar representations to use
    simInfo : pd.DataFrame
        DataFrame with hierarchical category information
    file_name : str
        Filename to save/load performance data
    seeds : list
        Random seeds to use for simulation
    
    Returns:
    --------
    tuple
        (subSimPerformance, subSimPerfAccSummary, subSimPerfRTSummary)
    """
    
    if not os.path.exists(file_name):

        simInfo = create_simulated_image_info()
        # Preallocate performance dataframe
        subSimPerformance = pd.DataFrame()

        for seed in seeds:
            # Set random seed for reproducibility
            rng = np.random.default_rng(seed)
            
            # Simulate task using the existing category structure
            perf = simulate_task(
                simInfo,
                rng=rng,
                exemplars=simReps,
                balanced=False,
                foils=False,
                ebrw_kwargs=ebrw_kwargs,
            )

            # Save to dataframe
            perf["seed"] = seed
            perf["model"] = "simulated"
            subSimPerformance = pd.concat([subSimPerformance, perf], axis=0)

        # Save
        subSimPerformance.to_csv(file_name, index=False)
    else:
        print("Loading performance from file")
        subSimPerformance = pd.read_csv(file_name)

    hierOrder = ["super", "basic", "sub"]
    subSimPerformance["level"] = pd.Categorical(
        subSimPerformance["level"], categories=hierOrder, ordered=True
    )

    # Summary
    subSimPerfAccSummary = (
        subSimPerformance.groupby(["seed", "level"])["response"]
        .agg(lambda x: np.mean(x == "yes"))
        .groupby(["level"])
        .agg(["mean", "std"])
    )
    subSimPerfRTSummary = (
        subSimPerformance.groupby(["seed", "level"])["RT"]
        .agg("mean")
        .groupby(["level"])
        .agg(["mean", "std"])
    )

    return subSimPerformance, subSimPerfAccSummary, subSimPerfRTSummary