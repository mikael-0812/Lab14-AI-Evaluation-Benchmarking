from typing import Any, Dict, Iterable, List, Optional
import math
import statistics


class RetrievalEvaluator:
    """Evaluate retrieval quality for the retrieval stage of a RAG pipeline."""

    def __init__(self):
        self.results = []

    def _normalize_doc_ids(self, doc_ids: Iterable[Any]) -> List[str]:
        """Convert different document-id shapes into a comparable string list."""
        normalized: List[str] = []

        for item in doc_ids or []:
            candidate = item
            if isinstance(item, dict):
                candidate = item.get("id") or item.get("doc_id") or item.get("document_id")

            if candidate is None:
                continue

            candidate_str = str(candidate).strip()
            if candidate_str:
                normalized.append(candidate_str)

        return normalized

    def calculate_hit_rate(self, expected_ids: List[str], retrieved_ids: List[str], top_k: int = 3) -> float:
        """
        Return 1.0 when any expected document appears inside the top-k results.

        Hit Rate is binary for each query:
        - 1.0 if at least one ground-truth document is retrieved in the top-k ranking
        - 0.0 otherwise
        """
        if top_k <= 0:
            return 0.0

        expected = set(self._normalize_doc_ids(expected_ids))
        top_retrieved = self._normalize_doc_ids(retrieved_ids)[:top_k]

        if not expected or not top_retrieved:
            return 0.0

        return 1.0 if any(doc_id in expected for doc_id in top_retrieved) else 0.0

    def calculate_retrieval_accuracy(
        self,
        expected_ids: List[str],
        retrieved_ids: List[str],
        top_k: int = 3,
    ) -> float:
        """
        Calculate how much of the ground-truth set is covered in the top-k retrieval results.
        """
        if top_k <= 0:
            return 0.0

        expected = set(self._normalize_doc_ids(expected_ids))
        top_retrieved = set(self._normalize_doc_ids(retrieved_ids)[:top_k])

        if not expected or not top_retrieved:
            return 0.0

        matched = len(expected.intersection(top_retrieved))
        return matched / len(expected)

    def calculate_mrr(self, expected_ids: List[str], retrieved_ids: List[str]) -> float:
        """
        Return the reciprocal rank of the first correct document in the ranking.

        Example:
        expected_ids = ["doc2"]
        retrieved_ids = ["doc1", "doc3", "doc2", "doc4"]
        => reciprocal rank = 1 / 3
        """
        expected = set(self._normalize_doc_ids(expected_ids))
        retrieved = self._normalize_doc_ids(retrieved_ids)

        if not expected or not retrieved:
            return 0.0

        for index, doc_id in enumerate(retrieved):
            if doc_id in expected:
                return 1.0 / (index + 1)

        return 0.0

    def calculate_ndcg(self, expected_ids: List[str], retrieved_ids: List[str], top_k: int = 5) -> float:
        """
        Calculate NDCG (Normalized Discounted Cumulative Gain).
        This rewards retrieving relevant documents earlier in the ranking.
        """
        if top_k <= 0:
            return 0.0

        expected = set(self._normalize_doc_ids(expected_ids))
        retrieved = self._normalize_doc_ids(retrieved_ids)

        if not expected or not retrieved:
            return 0.0

        dcg = 0.0
        counted_relevant = set()
        for index, doc_id in enumerate(retrieved[:top_k]):
            if doc_id in expected and doc_id not in counted_relevant:
                dcg += 1.0 / math.log2(index + 2)
                counted_relevant.add(doc_id)

        idcg = sum(1.0 / math.log2(index + 2) for index in range(min(len(expected), top_k)))
        return dcg / idcg if idcg > 0 else 0.0

    async def score(self, test_case: Dict, agent_response: Dict) -> Dict:
        """
        Compute retrieval metrics for a single test case.

        Args:
            test_case: Contains expected_retrieval_ids (ground truth)
            agent_response: Contains retrieved_ids returned by the agent

        Returns:
            Dict with hit_rate, retrieval_accuracy, mrr, ndcg
        """
        expected_ids = self._normalize_doc_ids(test_case.get("expected_retrieval_ids", []))
        retrieved_ids = self._normalize_doc_ids(agent_response.get("retrieved_ids", []))

        hit_rate = self.calculate_hit_rate(expected_ids, retrieved_ids, top_k=3)
        retrieval_accuracy = self.calculate_retrieval_accuracy(expected_ids, retrieved_ids, top_k=3)
        mrr = self.calculate_mrr(expected_ids, retrieved_ids)
        ndcg = self.calculate_ndcg(expected_ids, retrieved_ids, top_k=5)

        return {
            "hit_rate": hit_rate,
            "retrieval_accuracy": retrieval_accuracy,
            "mrr": mrr,
            "ndcg": ndcg,
            "retrieved_count": len(retrieved_ids),
            "expected_count": len(expected_ids),
        }

    async def evaluate_batch(self, dataset: List[Dict], agent_responses: Optional[List[Dict]] = None) -> Dict:
        """
        Run retrieval evaluation for a dataset and return aggregate metrics.
        """
        if agent_responses is None:
            agent_responses = [
                {"retrieved_ids": test_case.get("retrieved_ids", [])}
                for test_case in dataset
            ]

        hit_rates = []
        retrieval_accuracies = []
        mrrs = []
        ndcgs = []

        for test_case, response in zip(dataset, agent_responses):
            metrics = await self.score(test_case, response)
            hit_rates.append(metrics["hit_rate"])
            retrieval_accuracies.append(metrics["retrieval_accuracy"])
            mrrs.append(metrics["mrr"])
            ndcgs.append(metrics["ndcg"])

        return {
            "avg_hit_rate": statistics.mean(hit_rates) if hit_rates else 0.0,
            "avg_retrieval_accuracy": statistics.mean(retrieval_accuracies) if retrieval_accuracies else 0.0,
            "avg_mrr": statistics.mean(mrrs) if mrrs else 0.0,
            "avg_ndcg": statistics.mean(ndcgs) if ndcgs else 0.0,
            "total_cases": len(dataset),
        }
