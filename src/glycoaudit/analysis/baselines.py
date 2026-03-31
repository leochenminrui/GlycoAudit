"""
Baseline Models for Glycan-Agent Binding Prediction

Implements four groups of baselines to demonstrate benchmark generality:
- Group 1: Null/Simple baselines (random, permutation, nearest neighbor, prototype)
- Group 2: Shallow ML baselines (linear, random forest, XGBoost)
- Group 3: Multi-prototype variants (k-means, mixture)
- Group 4: Sequence/motif baselines (if sequence data available)

All baselines implement a common interface for fair comparison.
"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Try to import optional dependencies
try:
    from sklearn.cluster import KMeans
    HAS_KMEANS = True
except ImportError:
    HAS_KMEANS = False

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False


class BaselineModel(ABC):
    """Abstract base class for baseline models."""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def fit(self, train_glycan_ids: List[str], train_positives: List[str],
            features: np.ndarray, glycan_to_idx: Dict[str, int]):
        """Train the model on positive examples."""
        pass

    @abstractmethod
    def predict(self, candidate_glycan_ids: List[str],
                features: np.ndarray, glycan_to_idx: Dict[str, int]) -> np.ndarray:
        """Predict scores for candidate glycans."""
        pass


# =============================================================================
# GROUP 1: NULL / SIMPLE BASELINES
# =============================================================================

class RandomBaseline(BaselineModel):
    """Random baseline - assigns random scores."""

    def __init__(self, seed: int = 42):
        super().__init__("Random")
        self.rng = np.random.RandomState(seed)

    def fit(self, train_glycan_ids: List[str], train_positives: List[str],
            features: np.ndarray, glycan_to_idx: Dict[str, int]):
        pass  # No training needed

    def predict(self, candidate_glycan_ids: List[str],
                features: np.ndarray, glycan_to_idx: Dict[str, int]) -> np.ndarray:
        return self.rng.rand(len(candidate_glycan_ids))


class PermutationBaseline(BaselineModel):
    """Permutation baseline - uses shuffled features."""

    def __init__(self, seed: int = 42):
        super().__init__("Permutation")
        self.rng = np.random.RandomState(seed)
        self.prototype = None

    def fit(self, train_glycan_ids: List[str], train_positives: List[str],
            features: np.ndarray, glycan_to_idx: Dict[str, int]):
        # Get positive indices
        pos_indices = [glycan_to_idx[gid] for gid in train_positives if gid in glycan_to_idx]
        if not pos_indices:
            self.prototype = np.zeros(features.shape[1])
            return

        # Get features for positives
        pos_features = features[pos_indices]

        # Shuffle each feature column independently
        shuffled_features = pos_features.copy()
        for j in range(shuffled_features.shape[1]):
            self.rng.shuffle(shuffled_features[:, j])

        # Compute mean of shuffled features
        self.prototype = shuffled_features.mean(axis=0)

    def predict(self, candidate_glycan_ids: List[str],
                features: np.ndarray, glycan_to_idx: Dict[str, int]) -> np.ndarray:
        scores = []
        for gid in candidate_glycan_ids:
            if gid not in glycan_to_idx:
                scores.append(0.0)
                continue

            idx = glycan_to_idx[gid]
            glycan_features = features[idx]

            # Cosine similarity
            norm_proto = np.linalg.norm(self.prototype)
            norm_glycan = np.linalg.norm(glycan_features)

            if norm_proto == 0 or norm_glycan == 0:
                scores.append(0.0)
            else:
                score = np.dot(self.prototype, glycan_features) / (norm_proto * norm_glycan)
                scores.append(float(score))

        return np.array(scores)


class NearestNeighborBaseline(BaselineModel):
    """Nearest neighbor baseline - score by max similarity to any positive."""

    def __init__(self, name: str = "NearestNeighbor", **kwargs):
        super().__init__(name)
        self.train_positives = None
        self.positive_features = None

    def fit(self, train_glycan_ids: List[str], train_positives: List[str],
            features: np.ndarray, glycan_to_idx: Dict[str, int]):
        self.train_positives = train_positives

        # Get features for positives
        pos_indices = [glycan_to_idx[gid] for gid in train_positives if gid in glycan_to_idx]
        if pos_indices:
            self.positive_features = features[pos_indices]
        else:
            self.positive_features = np.zeros((0, features.shape[1]))

    def predict(self, candidate_glycan_ids: List[str],
                features: np.ndarray, glycan_to_idx: Dict[str, int]) -> np.ndarray:
        if len(self.positive_features) == 0:
            return np.zeros(len(candidate_glycan_ids))

        scores = []
        for gid in candidate_glycan_ids:
            if gid not in glycan_to_idx:
                scores.append(0.0)
                continue

            idx = glycan_to_idx[gid]
            glycan_features = features[idx]

            # Compute cosine similarity to all positives
            similarities = []
            for pos_feat in self.positive_features:
                norm_pos = np.linalg.norm(pos_feat)
                norm_glycan = np.linalg.norm(glycan_features)

                if norm_pos == 0 or norm_glycan == 0:
                    similarities.append(0.0)
                else:
                    sim = np.dot(pos_feat, glycan_features) / (norm_pos * norm_glycan)
                    similarities.append(float(sim))

            # Take max similarity
            scores.append(max(similarities))

        return np.array(scores)


class PrototypeBaseline(BaselineModel):
    """Prototype baseline - mean of positives (current method)."""

    def __init__(self, **kwargs):
        super().__init__("Prototype")
        self.prototype = None

    def fit(self, train_glycan_ids: List[str], train_positives: List[str],
            features: np.ndarray, glycan_to_idx: Dict[str, int]):
        pos_indices = [glycan_to_idx[gid] for gid in train_positives if gid in glycan_to_idx]
        if not pos_indices:
            self.prototype = np.zeros(features.shape[1])
            return

        pos_features = features[pos_indices]
        self.prototype = pos_features.mean(axis=0)

    def predict(self, candidate_glycan_ids: List[str],
                features: np.ndarray, glycan_to_idx: Dict[str, int]) -> np.ndarray:
        scores = []
        for gid in candidate_glycan_ids:
            if gid not in glycan_to_idx:
                scores.append(0.0)
                continue

            idx = glycan_to_idx[gid]
            glycan_features = features[idx]

            # Cosine similarity
            norm_proto = np.linalg.norm(self.prototype)
            norm_glycan = np.linalg.norm(glycan_features)

            if norm_proto == 0 or norm_glycan == 0:
                scores.append(0.0)
            else:
                score = np.dot(self.prototype, glycan_features) / (norm_proto * norm_glycan)
                scores.append(float(score))

        return np.array(scores)


# =============================================================================
# GROUP 2: SHALLOW ML BASELINES
# =============================================================================

class LinearBaseline(BaselineModel):
    """Linear baseline - logistic regression with balanced class weights."""

    def __init__(self, seed: int = 42):
        super().__init__("Linear")
        self.model = LogisticRegression(random_state=seed, max_iter=1000,
                                        class_weight='balanced')
        self.train_glycan_ids = None

    def fit(self, train_glycan_ids: List[str], train_positives: List[str],
            features: np.ndarray, glycan_to_idx: Dict[str, int]):
        self.train_glycan_ids = train_glycan_ids

        # Create labels: 1 for positives, 0 for unlabeled
        train_indices = [glycan_to_idx[gid] for gid in train_glycan_ids if gid in glycan_to_idx]
        X_train = features[train_indices]

        y_train = np.array([1 if gid in train_positives else 0 for gid in train_glycan_ids
                           if gid in glycan_to_idx])

        # Fit model
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model.fit(X_train, y_train)

    def predict(self, candidate_glycan_ids: List[str],
                features: np.ndarray, glycan_to_idx: Dict[str, int]) -> np.ndarray:
        candidate_indices = [glycan_to_idx.get(gid, -1) for gid in candidate_glycan_ids]

        scores = []
        for idx in candidate_indices:
            if idx == -1:
                scores.append(0.0)
            else:
                X = features[idx:idx+1]
                score = self.model.predict_proba(X)[0, 1]
                scores.append(float(score))

        return np.array(scores)


class RandomForestBaseline(BaselineModel):
    """Random Forest baseline."""

    def __init__(self, n_estimators: int = 100, seed: int = 42):
        super().__init__("RandomForest")
        self.model = RandomForestClassifier(n_estimators=n_estimators,
                                           random_state=seed,
                                           max_depth=10,
                                           class_weight='balanced')
        self.train_glycan_ids = None

    def fit(self, train_glycan_ids: List[str], train_positives: List[str],
            features: np.ndarray, glycan_to_idx: Dict[str, int]):
        self.train_glycan_ids = train_glycan_ids

        train_indices = [glycan_to_idx[gid] for gid in train_glycan_ids if gid in glycan_to_idx]
        X_train = features[train_indices]

        y_train = np.array([1 if gid in train_positives else 0 for gid in train_glycan_ids
                           if gid in glycan_to_idx])

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model.fit(X_train, y_train)

    def predict(self, candidate_glycan_ids: List[str],
                features: np.ndarray, glycan_to_idx: Dict[str, int]) -> np.ndarray:
        candidate_indices = [glycan_to_idx.get(gid, -1) for gid in candidate_glycan_ids]

        scores = []
        for idx in candidate_indices:
            if idx == -1:
                scores.append(0.0)
            else:
                X = features[idx:idx+1]
                score = self.model.predict_proba(X)[0, 1]
                scores.append(float(score))

        return np.array(scores)


# =============================================================================
# GROUP 3: MULTI-PROTOTYPE VARIANTS
# =============================================================================

class MultiPrototypeBaseline(BaselineModel):
    """Multi-prototype baseline using k-means clustering."""

    def __init__(self, n_prototypes: int = 3, seed: int = 42):
        super().__init__(f"MultiPrototype_k{n_prototypes}")
        self.n_prototypes = n_prototypes
        self.seed = seed
        self.prototypes = None

        if not HAS_KMEANS:
            raise ImportError("sklearn KMeans required for MultiPrototypeBaseline")

    def fit(self, train_glycan_ids: List[str], train_positives: List[str],
            features: np.ndarray, glycan_to_idx: Dict[str, int]):
        pos_indices = [glycan_to_idx[gid] for gid in train_positives if gid in glycan_to_idx]

        if len(pos_indices) < self.n_prototypes:
            # Fall back to single prototype
            if pos_indices:
                self.prototypes = [features[pos_indices].mean(axis=0)]
            else:
                self.prototypes = [np.zeros(features.shape[1])]
            return

        pos_features = features[pos_indices]

        # Cluster positives
        kmeans = KMeans(n_clusters=self.n_prototypes, random_state=self.seed, n_init=10)
        kmeans.fit(pos_features)

        self.prototypes = kmeans.cluster_centers_

    def predict(self, candidate_glycan_ids: List[str],
                features: np.ndarray, glycan_to_idx: Dict[str, int]) -> np.ndarray:
        scores = []
        for gid in candidate_glycan_ids:
            if gid not in glycan_to_idx:
                scores.append(0.0)
                continue

            idx = glycan_to_idx[gid]
            glycan_features = features[idx]

            # Max similarity to any prototype
            max_sim = 0.0
            for proto in self.prototypes:
                norm_proto = np.linalg.norm(proto)
                norm_glycan = np.linalg.norm(glycan_features)

                if norm_proto > 0 and norm_glycan > 0:
                    sim = np.dot(proto, glycan_features) / (norm_proto * norm_glycan)
                    max_sim = max(max_sim, float(sim))

            scores.append(max_sim)

        return np.array(scores)


# =============================================================================
# BASELINE FACTORY
# =============================================================================

def get_baseline_model(baseline_name: str, **kwargs) -> BaselineModel:
    """Factory function to create baseline models."""
    baseline_map = {
        "random": RandomBaseline,
        "permutation": PermutationBaseline,
        "nearest_neighbor": NearestNeighborBaseline,
        "prototype": PrototypeBaseline,
        "linear": LinearBaseline,
        "random_forest": RandomForestBaseline,
        "multi_prototype": MultiPrototypeBaseline,
    }

    if baseline_name.lower() not in baseline_map:
        raise ValueError(f"Unknown baseline: {baseline_name}. "
                        f"Available: {list(baseline_map.keys())}")

    baseline_class = baseline_map[baseline_name.lower()]
    return baseline_class(**kwargs)


def get_available_baselines() -> Dict[str, str]:
    """Get dictionary of available baselines with descriptions."""
    baselines = {
        "random": "Random scores (no structure information)",
        "permutation": "Prototype with shuffled features (null model)",
        "nearest_neighbor": "Max similarity to any positive (1-NN)",
        "prototype": "Mean of positives + cosine similarity (current method)",
        "linear": "Logistic regression (balanced)",
        "random_forest": "Random Forest (balanced)",
        "multi_prototype": "k-means clustering of positives",
    }

    return baselines
