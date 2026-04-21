import asyncio
import inspect
import time
from typing import Dict, List


class BenchmarkRunner:
    """Async benchmark runner with retrieval, judge, and audit support."""

    def __init__(self, agent, retrieval_evaluator, llm_judge):
        self.agent = agent
        self.retrieval_evaluator = retrieval_evaluator
        self.llm_judge = llm_judge
        self.results = []
        self.estimated_agent_cost_per_1k = 0.00015
        self._agent_supports_test_case = "test_case" in inspect.signature(self.agent.query).parameters

    async def _query_agent(self, test_case: Dict) -> Dict:
        if self._agent_supports_test_case:
            return await self.agent.query(test_case["question"], test_case=test_case)
        return await self.agent.query(test_case["question"])

    async def run_single_test(self, test_case: Dict) -> Dict:
        start_time = time.perf_counter()

        try:
            response = await self._query_agent(test_case)
            agent_latency = time.perf_counter() - start_time

            retrieval_scores = await self.retrieval_evaluator.score(test_case, response)
            judge_result = await self.llm_judge.evaluate_multi_judge(
                test_case["question"],
                response.get("answer", ""),
                test_case.get("expected_answer", ""),
            )

            total_latency = time.perf_counter() - start_time
            agent_tokens = response.get("metadata", {}).get("tokens_used", 0)
            judge_tokens = judge_result.get("judge_tokens_used", 0)
            estimated_agent_cost_usd = round((agent_tokens / 1000) * self.estimated_agent_cost_per_1k, 6)
            estimated_total_cost_usd = round(
                estimated_agent_cost_usd + judge_result.get("estimated_judge_cost_usd", 0.0),
                6,
            )

            is_pass = (
                judge_result["final_score"] >= 3.0
                and retrieval_scores["hit_rate"] >= 0.5
                and judge_result.get("flags", {}).get("merged", {}).get("hallucination", 0) == 0
            )

            result = {
                "question": test_case["question"],
                "agent_response": response.get("answer", ""),
                "expected_answer": test_case.get("expected_answer", ""),
                "latency_seconds": round(agent_latency, 3),
                "total_latency_seconds": round(total_latency, 3),
                "retrieval_metrics": retrieval_scores,
                "judge_result": judge_result,
                "tokens_used": agent_tokens,
                "judge_tokens_used": judge_tokens,
                "estimated_agent_cost_usd": estimated_agent_cost_usd,
                "estimated_total_cost_usd": estimated_total_cost_usd,
                "status": "pass" if is_pass else "fail",
            }

            return result
        except Exception as exc:
            print(f"Error in test case '{test_case.get('question', 'unknown')}': {exc}")
            return {
                "question": test_case.get("question", "unknown"),
                "agent_response": "",
                "expected_answer": test_case.get("expected_answer", ""),
                "latency_seconds": 0,
                "total_latency_seconds": 0,
                "retrieval_metrics": {"hit_rate": 0, "retrieval_accuracy": 0, "mrr": 0, "ndcg": 0},
                "judge_result": {
                    "final_score": 1,
                    "agreement_rate": 0,
                    "verdict": "incorrect",
                    "flags": {"merged": {"hallucination": 1, "bias": 0, "partial_correct": 0}},
                    "dimension_scores": {"merged": {"accuracy": 1, "tone": 1, "fairness": 1, "consistency": 1}},
                    "user_satisfaction_score": 1,
                    "manual_review_recommended": True,
                    "review_reasons": ["runner_exception"],
                },
                "tokens_used": 0,
                "judge_tokens_used": 0,
                "estimated_agent_cost_usd": 0.0,
                "estimated_total_cost_usd": 0.0,
                "status": "error",
                "error": str(exc),
            }

    async def run_all(self, dataset: List[Dict], batch_size: int = 3) -> List[Dict]:
        self.results = []
        total_batches = (len(dataset) + batch_size - 1) // batch_size

        print(f"Running {len(dataset)} tests in {total_batches} batches (batch_size={batch_size})...")

        for batch_idx in range(0, len(dataset), batch_size):
            batch = dataset[batch_idx : batch_idx + batch_size]
            print(f"\nBatch {batch_idx // batch_size + 1}/{total_batches} ({len(batch)} cases)")

            tasks = [self.run_single_test(case) for case in batch]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)

            for result in batch_results:
                if isinstance(result, Exception):
                    print(f"Batch error: {result}")
                    continue

                self.results.append(result)
                status = result["status"]
                status_label = "PASS" if status == "pass" else "ERROR" if status == "error" else "FAIL"
                print(f"{status_label} {result['question'][:50]}... (score: {result['judge_result'].get('final_score', 'N/A')})")

            if batch_idx + batch_size < len(dataset):
                await asyncio.sleep(1)

        return self.results

    def get_audit_candidates(self, limit: int = 10) -> List[Dict]:
        def priority(result: Dict) -> tuple:
            judge_result = result.get("judge_result", {})
            reasons = judge_result.get("review_reasons", [])
            return (
                len(reasons),
                1 if judge_result.get("manual_review_recommended") else 0,
                -judge_result.get("agreement_rate", 0),
                -judge_result.get("flags", {}).get("merged", {}).get("hallucination", 0),
                -judge_result.get("final_score", 0),
            )

        flagged = [
            result
            for result in self.results
            if result.get("judge_result", {}).get("manual_review_recommended")
            or result.get("judge_result", {}).get("agreement_rate", 1) < 0.75
            or result.get("judge_result", {}).get("flags", {}).get("merged", {}).get("hallucination", 0) == 1
        ]

        flagged.sort(key=priority, reverse=True)
        return flagged[:limit]

    def get_summary(self) -> Dict:
        if not self.results:
            return {"error": "No results available"}

        valid_results = [result for result in self.results if result.get("status") != "error"]
        pass_count = sum(1 for result in self.results if result.get("status") == "pass")
        fail_count = sum(1 for result in self.results if result.get("status") == "fail")
        error_count = sum(1 for result in self.results if result.get("status") == "error")

        judge_scores = [result["judge_result"].get("final_score", 3) for result in valid_results]
        hit_rates = [result["retrieval_metrics"].get("hit_rate", 0) for result in valid_results]
        retrieval_accuracies = [result["retrieval_metrics"].get("retrieval_accuracy", 0) for result in valid_results]
        mrrs = [result["retrieval_metrics"].get("mrr", 0) for result in valid_results]
        latencies = [result.get("total_latency_seconds", 0) for result in valid_results]
        agreement_rates = [result["judge_result"].get("agreement_rate", 0) for result in valid_results]
        answer_accuracies = [
            result["judge_result"].get("dimension_scores", {}).get("merged", {}).get("accuracy", 0)
            for result in valid_results
        ]
        fairness_scores = [
            result["judge_result"].get("dimension_scores", {}).get("merged", {}).get("fairness", 0)
            for result in valid_results
        ]
        consistency_scores = [
            result["judge_result"].get("dimension_scores", {}).get("merged", {}).get("consistency", 0)
            for result in valid_results
        ]
        user_satisfaction_scores = [
            result["judge_result"].get("user_satisfaction_score", 0)
            for result in valid_results
        ]
        hallucination_flags = [
            result["judge_result"].get("flags", {}).get("merged", {}).get("hallucination", 0)
            for result in valid_results
        ]
        bias_flags = [
            result["judge_result"].get("flags", {}).get("merged", {}).get("bias", 0)
            for result in valid_results
        ]
        partial_correct_flags = [
            result["judge_result"].get("flags", {}).get("merged", {}).get("partial_correct", 0)
            for result in valid_results
        ]
        tokens_used = [result.get("tokens_used", 0) + result.get("judge_tokens_used", 0) for result in valid_results]
        total_costs = [result.get("estimated_total_cost_usd", 0.0) for result in valid_results]
        manual_review_count = sum(
            1 for result in valid_results if result["judge_result"].get("manual_review_recommended")
        )

        result_count = len(self.results) or 1
        valid_count = len(valid_results) or 1

        return {
            "total_cases": len(self.results),
            "pass": pass_count,
            "fail": fail_count,
            "error": error_count,
            "pass_rate": round(pass_count / result_count * 100, 2),
            "avg_judge_score": round(sum(judge_scores) / valid_count, 2) if judge_scores else 0,
            "avg_hit_rate": round(sum(hit_rates) / valid_count, 2) if hit_rates else 0,
            "avg_retrieval_accuracy": round(sum(retrieval_accuracies) / valid_count, 2) if retrieval_accuracies else 0,
            "avg_mrr": round(sum(mrrs) / valid_count, 2) if mrrs else 0,
            "avg_latency_seconds": round(sum(latencies) / valid_count, 2) if latencies else 0,
            "avg_agreement_rate": round(sum(agreement_rates) / valid_count, 2) if agreement_rates else 0,
            "avg_final_answer_accuracy": round(sum(answer_accuracies) / valid_count, 2) if answer_accuracies else 0,
            "avg_fairness_score": round(sum(fairness_scores) / valid_count, 2) if fairness_scores else 0,
            "avg_consistency_score": round(sum(consistency_scores) / valid_count, 2) if consistency_scores else 0,
            "hallucination_rate": round(sum(hallucination_flags) / valid_count, 2) if hallucination_flags else 0,
            "bias_rate": round(sum(bias_flags) / valid_count, 2) if bias_flags else 0,
            "partial_correct_rate": round(sum(partial_correct_flags) / valid_count, 2) if partial_correct_flags else 0,
            "user_satisfaction_score": round(sum(user_satisfaction_scores) / valid_count, 2)
            if user_satisfaction_scores
            else 0,
            "avg_tokens_used": round(sum(tokens_used) / valid_count, 2) if tokens_used else 0,
            "total_tokens_used": int(sum(tokens_used)) if tokens_used else 0,
            "estimated_total_cost_usd": round(sum(total_costs), 6) if total_costs else 0.0,
            "estimated_avg_cost_usd": round(sum(total_costs) / valid_count, 6) if total_costs else 0.0,
            "manual_review_rate": round(manual_review_count / valid_count, 2) if valid_results else 0,
        }
