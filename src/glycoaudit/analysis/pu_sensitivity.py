"""
PU (Positive-Unlabeled) Sensitivity Analysis

Analyzes sensitivity of ranking metrics to positive class prior assumptions
and potential contamination of the unlabeled set.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd


@dataclass
class PUSensitivityResult:
    """
    Result of PU sensitivity analysis.

    Attributes:
        agent_id: Agent identifier
        n_labeled_pos: Number of labeled positives
        n_candidates: Total number of candidates
        observed_metrics: Observed metrics with all unlabeled as negative
        sensitivity_curves: Metrics under different contamination assumptions
        pi_estimates: Estimated positive class priors
        metadata: Additional metadata
    """
    agent_id: str
    n_labeled_pos: int
    n_candidates: int
    observed_metrics: Dict[str, float]
    sensitivity_curves: List[Dict]
    pi_estimates: Dict[str, float]
    metadata: Optional[Dict] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'agent_id': self.agent_id,
            'n_labeled_pos': self.n_labeled_pos,
            'n_candidates': self.n_candidates,
            'observed_metrics': self.observed_metrics,
            'sensitivity_curves': self.sensitivity_curves,
            'pi_estimates': self.pi_estimates,
            'metadata': self.metadata or {}
        }


class PUSensitivityAnalyzer:
    """
    Analyzer for PU learning sensitivity under contamination assumptions.
    """

    def __init__(self, seed: int = 42):
        """
        Initialize analyzer.

        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def estimate_class_prior(
        self,
        n_labeled_pos: int,
        n_candidates: int,
        method: str = 'pessimistic'
    ) -> float:
        """
        Estimate positive class prior π.

        Args:
            n_labeled_pos: Number of known positives
            n_candidates: Total number of candidates
            method: Estimation method ('optimistic', 'pessimistic', 'conservative')

        Returns:
            Estimated prior probability
        """
        if method == 'optimistic':
            # Assume labeled positives are ~80% of all positives
            return n_labeled_pos / (0.8 * n_candidates)

        elif method == 'pessimistic':
            # Assume labeled positives are ~50% of all positives
            return n_labeled_pos / (0.5 * n_candidates)

        elif method == 'conservative':
            # Assume labeled positives are ~20% of all positives
            return n_labeled_pos / (0.2 * n_candidates)

        elif method == 'observed':
            # Use observed proportion (lower bound)
            return n_labeled_pos / n_candidates

        else:
            raise ValueError(f"Unknown method: {method}")

    def analyze_contamination_sensitivity(
        self,
        scores: np.ndarray,
        labeled_pos_indices: List[int],
        contamination_levels: List[float] = None,
        n_bootstrap: int = 100
    ) -> List[Dict]:
        """
        Analyze how metrics change under different contamination assumptions.

        Contamination = fraction of top-ranked unlabeled that are actually positive.

        Args:
            scores: Ranking scores for all candidates
            labeled_pos_indices: Indices of labeled positives
            contamination_levels: List of contamination fractions to test
            n_bootstrap: Number of bootstrap samples per level

        Returns:
            List of dictionaries with metrics per contamination level
        """
        if contamination_levels is None:
            contamination_levels = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5]

        n_candidates = len(scores)
        n_labeled_pos = len(labeled_pos_indices)

        # Rank all candidates
        sorted_indices = np.argsort(-scores)
        rank_of_idx = np.empty(n_candidates, dtype=int)
        rank_of_idx[sorted_indices] = np.arange(1, n_candidates + 1)

        # Observed ranks of labeled positives
        labeled_pos_ranks = [rank_of_idx[idx] for idx in labeled_pos_indices]

        # Unlabeled indices
        labeled_pos_set = set(labeled_pos_indices)
        unlabeled_indices = [i for i in range(n_candidates) if i not in labeled_pos_set]

        results = []

        for contam in contamination_levels:
            # Estimate number of hidden positives in top-ranked unlabeled
            n_top_unlabeled = min(n_labeled_pos * 2, len(unlabeled_indices))
            n_contaminated = int(n_top_unlabeled * contam)

            # Bootstrap to estimate metric variability
            bootstrap_metrics = []

            for _ in range(n_bootstrap):
                # Sample which unlabeled are actually positive
                if n_contaminated > 0:
                    # Get top-ranked unlabeled
                    unlabeled_ranks = [(rank_of_idx[idx], idx) for idx in unlabeled_indices]
                    unlabeled_ranks.sort()
                    top_unlabeled_indices = [idx for _, idx in unlabeled_ranks[:n_top_unlabeled]]

                    # Randomly select contaminated
                    contaminated_indices = self.rng.choice(
                        top_unlabeled_indices,
                        size=n_contaminated,
                        replace=False
                    ).tolist()
                else:
                    contaminated_indices = []

                # Combined positives
                all_pos_indices = labeled_pos_indices + contaminated_indices
                all_pos_ranks = [rank_of_idx[idx] for idx in all_pos_indices]

                # Compute metrics
                if all_pos_ranks:
                    mrr = 1.0 / min(all_pos_ranks)
                    recall_5 = sum(r <= 5 for r in all_pos_ranks) / len(all_pos_ranks)
                    recall_10 = sum(r <= 10 for r in all_pos_ranks) / len(all_pos_ranks)
                    mean_rank = np.mean(all_pos_ranks)
                else:
                    mrr = recall_5 = recall_10 = mean_rank = 0.0

                bootstrap_metrics.append({
                    'mrr': mrr,
                    'recall@5': recall_5,
                    'recall@10': recall_10,
                    'mean_rank': mean_rank
                })

            # Aggregate bootstrap results
            df_boot = pd.DataFrame(bootstrap_metrics)

            result = {
                'contamination': contam,
                'n_contaminated': n_contaminated,
                'n_total_positives': n_labeled_pos + n_contaminated,
                'mrr_mean': df_boot['mrr'].mean(),
                'mrr_std': df_boot['mrr'].std(),
                'mrr_ci_lower': df_boot['mrr'].quantile(0.025),
                'mrr_ci_upper': df_boot['mrr'].quantile(0.975),
                'recall@5_mean': df_boot['recall@5'].mean(),
                'recall@5_std': df_boot['recall@5'].std(),
                'recall@10_mean': df_boot['recall@10'].mean(),
                'recall@10_std': df_boot['recall@10'].std(),
                'mean_rank_mean': df_boot['mean_rank'].mean(),
            }

            results.append(result)

        return results

    def analyze_agent(
        self,
        agent_id: str,
        scores: np.ndarray,
        labeled_pos_indices: List[int],
        contamination_levels: List[float] = None,
        n_bootstrap: int = 100
    ) -> PUSensitivityResult:
        """
        Perform full PU sensitivity analysis for a single agent.

        Args:
            agent_id: Agent identifier
            scores: Ranking scores for all candidates
            labeled_pos_indices: Indices of labeled positives
            contamination_levels: Contamination levels to test
            n_bootstrap: Number of bootstrap samples

        Returns:
            PUSensitivityResult object
        """
        n_candidates = len(scores)
        n_labeled_pos = len(labeled_pos_indices)

        # Compute observed metrics (treating all unlabeled as negative)
        sorted_indices = np.argsort(-scores)
        rank_of_idx = np.empty(n_candidates, dtype=int)
        rank_of_idx[sorted_indices] = np.arange(1, n_candidates + 1)

        labeled_pos_ranks = [rank_of_idx[idx] for idx in labeled_pos_indices]

        observed_metrics = {
            'mrr': 1.0 / min(labeled_pos_ranks) if labeled_pos_ranks else 0.0,
            'recall@5': sum(r <= 5 for r in labeled_pos_ranks) / len(labeled_pos_ranks) if labeled_pos_ranks else 0.0,
            'recall@10': sum(r <= 10 for r in labeled_pos_ranks) / len(labeled_pos_ranks) if labeled_pos_ranks else 0.0,
            'mean_rank': float(np.mean(labeled_pos_ranks)) if labeled_pos_ranks else 0.0,
        }

        # Estimate class priors
        pi_estimates = {
            'observed': self.estimate_class_prior(n_labeled_pos, n_candidates, 'observed'),
            'optimistic': self.estimate_class_prior(n_labeled_pos, n_candidates, 'optimistic'),
            'pessimistic': self.estimate_class_prior(n_labeled_pos, n_candidates, 'pessimistic'),
            'conservative': self.estimate_class_prior(n_labeled_pos, n_candidates, 'conservative'),
        }

        # Sensitivity analysis
        sensitivity_curves = self.analyze_contamination_sensitivity(
            scores,
            labeled_pos_indices,
            contamination_levels,
            n_bootstrap
        )

        return PUSensitivityResult(
            agent_id=agent_id,
            n_labeled_pos=n_labeled_pos,
            n_candidates=n_candidates,
            observed_metrics=observed_metrics,
            sensitivity_curves=sensitivity_curves,
            pi_estimates=pi_estimates,
            metadata={'n_bootstrap': n_bootstrap}
        )

    def aggregate_results(
        self,
        results: List[PUSensitivityResult]
    ) -> pd.DataFrame:
        """
        Aggregate sensitivity results across agents.

        Args:
            results: List of PUSensitivityResult objects

        Returns:
            DataFrame with aggregated metrics
        """
        aggregated = []

        # Get all contamination levels
        if results:
            contamination_levels = [c['contamination'] for c in results[0].sensitivity_curves]
        else:
            return pd.DataFrame()

        for contam in contamination_levels:
            contam_metrics = []

            for result in results:
                curve_point = next(
                    (c for c in result.sensitivity_curves if c['contamination'] == contam),
                    None
                )
                if curve_point:
                    contam_metrics.append(curve_point)

            if contam_metrics:
                df_contam = pd.DataFrame(contam_metrics)

                aggregated.append({
                    'contamination': contam,
                    'n_agents': len(contam_metrics),
                    'mrr_mean': df_contam['mrr_mean'].mean(),
                    'mrr_mean_std': df_contam['mrr_mean'].std(),
                    'recall@5_mean': df_contam['recall@5_mean'].mean(),
                    'recall@5_mean_std': df_contam['recall@5_mean'].std(),
                    'recall@10_mean': df_contam['recall@10_mean'].mean(),
                    'recall@10_mean_std': df_contam['recall@10_mean'].std(),
                })

        return pd.DataFrame(aggregated)

    def save_results(
        self,
        results: List[PUSensitivityResult],
        output_dir: Path
    ):
        """
        Save sensitivity analysis results.

        Args:
            results: List of PUSensitivityResult objects
            output_dir: Output directory
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save per-agent results
        per_agent_data = []
        for result in results:
            per_agent_data.append({
                'agent_id': result.agent_id,
                'n_labeled_pos': result.n_labeled_pos,
                'n_candidates': result.n_candidates,
                'observed_mrr': result.observed_metrics['mrr'],
                'observed_recall@5': result.observed_metrics['recall@5'],
                'observed_recall@10': result.observed_metrics['recall@10'],
                'pi_observed': result.pi_estimates['observed'],
                'pi_optimistic': result.pi_estimates['optimistic'],
                'pi_pessimistic': result.pi_estimates['pessimistic'],
            })

        df_per_agent = pd.DataFrame(per_agent_data)
        df_per_agent.to_csv(output_dir / 'per_agent_pi_estimates.csv', index=False)

        # Save sensitivity curves per agent
        all_curves = []
        for result in results:
            for curve_point in result.sensitivity_curves:
                all_curves.append({
                    'agent_id': result.agent_id,
                    **curve_point
                })

        df_curves = pd.DataFrame(all_curves)
        df_curves.to_csv(output_dir / 'sensitivity_curves.csv', index=False)

        # Save aggregated results
        df_agg = self.aggregate_results(results)
        df_agg.to_csv(output_dir / 'aggregated_sensitivity.csv', index=False)

        # Save summary
        summary = {
            'n_agents': len(results),
            'n_candidates': results[0].n_candidates if results else 0,
            'contamination_levels': [c['contamination'] for c in results[0].sensitivity_curves] if results else [],
            'method': 'Bootstrap contamination sensitivity analysis',
            'description': 'Evaluates metric sensitivity to hidden positives in unlabeled set'
        }

        with open(output_dir / 'summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
