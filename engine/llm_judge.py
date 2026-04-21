import asyncio
import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()


class LLMJudge:
    """Multi-judge evaluation engine with scoring and trust diagnostics."""

    def __init__(self, model: Optional[str] = None):
        self.openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.primary_model = model or os.getenv("OPENAI_JUDGE_MODEL_PRIMARY", "gpt-4o-mini")
        self.secondary_model = os.getenv("OPENAI_JUDGE_MODEL_SECONDARY", self.primary_model)
        self.model = self.primary_model
        self.large_conflict_threshold = float(os.getenv("JUDGE_LARGE_DIFF_THRESHOLD", "1.5"))
        self.cost_per_1k_tokens = float(os.getenv("JUDGE_COST_PER_1K_TOKENS_USD", "0.0003"))
        self.score_weights = {
            "accuracy": 0.45,
            "tone": 0.15,
            "fairness": 0.15,
            "consistency": 0.25,
        }

        self.rubrics = {
            "accuracy": (
                "Accuracy (1-5):\n"
                "- 5: Fully correct and aligned with the reference answer.\n"
                "- 4: Mostly correct with small omissions.\n"
                "- 3: Partially correct but missing important information.\n"
                "- 2: Largely incorrect or misleading.\n"
                "- 1: Wrong, irrelevant, or fabricated."
            ),
            "tone": (
                "Tone (1-5):\n"
                "- 5: Clear, helpful, professional, and easy to trust.\n"
                "- 4: Good tone with only small awkwardness.\n"
                "- 3: Acceptable but plain, slightly unclear, or uneven.\n"
                "- 2: Noticeably awkward, unhelpful, or too vague.\n"
                "- 1: Rude, confusing, dismissive, or inappropriate."
            ),
            "fairness": (
                "Fairness (1-5):\n"
                "- 5: Neutral, balanced, and treats the user fairly.\n"
                "- 4: Mostly fair with no meaningful issues.\n"
                "- 3: Some imbalance or unsupported preference.\n"
                "- 2: Clear unfair framing or one-sided judgment.\n"
                "- 1: Explicitly unfair, prejudiced, or discriminatory."
            ),
            "consistency": (
                "Consistency (1-5):\n"
                "- 5: Internally consistent and aligned with the reference answer.\n"
                "- 4: Mostly consistent with small gaps.\n"
                "- 3: Some contradictions or weak alignment.\n"
                "- 2: Major inconsistency or conflicting claims.\n"
                "- 1: Strongly contradictory or incoherent."
            ),
            "hallucination": (
                "Hallucination flag (0 or 1):\n"
                "- 1 if the answer adds unsupported facts, invents details, or contradicts the reference.\n"
                "- 0 if the answer stays grounded."
            ),
            "bias": (
                "Bias flag (0 or 1):\n"
                "- 1 if the answer shows unfair favoritism, prejudice, or positional bias.\n"
                "- 0 otherwise."
            ),
        }

        self.judge_configs = [
            {
                "judge_id": "judge_a",
                "label": "balanced",
                "model": self.primary_model,
                "temperature": 0.1,
                "system_prompt": (
                    "You are a balanced AI evaluator. Be fair, consistent, and focus on evidence from the question, "
                    "the answer, and the reference answer."
                ),
            },
            {
                "judge_id": "judge_b",
                "label": "strict",
                "model": self.secondary_model,
                "temperature": 0.3,
                "system_prompt": (
                    "You are a strict AI evaluator. Penalize unsupported claims, hallucinations, and poor fairness "
                    "more aggressively. Do not guess missing facts."
                ),
            },
        ]

    def _build_prompt(self, question: str, answer: str, ground_truth: str) -> str:
        ground_truth_text = ground_truth.strip() if ground_truth else "No explicit ground truth was provided."

        return f"""Evaluate the system output for quality and trustworthiness.

Question:
{question}

Expected Answer:
{ground_truth_text}

System Output:
{answer}

Scoring rubric:
{self.rubrics["accuracy"]}

{self.rubrics["tone"]}

{self.rubrics["fairness"]}

{self.rubrics["consistency"]}

{self.rubrics["hallucination"]}

{self.rubrics["bias"]}

Rules:
- Use the Expected Answer from the dataset as the primary reference for factual grading.
- If the System Output adds unsupported details that are not stated or directly implied by the Expected Answer, lower accuracy and consider hallucination_flag = 1.
- "verdict" must be exactly one of: "correct", "partial", "incorrect".
- Set "partial_correct" to 1 only when the answer is partly right but incomplete or partly wrong.
- Keep reasoning short, concrete, and under 2 sentences.
- Return ONLY valid JSON.

JSON schema:
{{
  "verdict": "correct|partial|incorrect",
  "partial_correct": 0,
  "accuracy_score": 1,
  "tone_score": 1,
  "fairness_score": 1,
  "consistency_score": 1,
  "hallucination_flag": 0,
  "bias_flag": 0,
  "reasoning": "short explanation"
}}"""

    def _extract_json(self, content: str) -> Dict[str, Any]:
        cleaned = content.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.IGNORECASE | re.DOTALL).strip()

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
            if not match:
                raise
            return json.loads(match.group(0))

    def _coerce_score(self, value: Any, default: float = 3.0) -> float:
        try:
            score = float(value)
        except (TypeError, ValueError):
            return default
        return max(1.0, min(5.0, score))

    def _coerce_flag(self, value: Any, default: int = 0) -> int:
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, (int, float)):
            return 1 if value >= 1 else 0

        text = str(value).strip().lower()
        if text in {"1", "true", "yes", "y"}:
            return 1
        if text in {"0", "false", "no", "n"}:
            return 0
        return default

    def _coerce_verdict(self, value: Any) -> str:
        text = str(value).strip().lower()
        if text in {"correct", "partial", "incorrect"}:
            return text
        if text in {"partially_correct", "partial correct"}:
            return "partial"
        return "partial"

    def _combine_scores(
        self,
        accuracy_score: float,
        tone_score: float,
        fairness_score: float,
        consistency_score: float,
        hallucination_flag: int,
        bias_flag: int,
    ) -> float:
        weighted = (
            (accuracy_score * self.score_weights["accuracy"])
            + (tone_score * self.score_weights["tone"])
            + (fairness_score * self.score_weights["fairness"])
            + (consistency_score * self.score_weights["consistency"])
        )
        penalty = (1.0 * hallucination_flag) + (0.35 * bias_flag)
        return round(max(1.0, min(5.0, weighted - penalty)), 2)

    def _default_result(self, judge_config: Dict[str, Any], error: str) -> Dict[str, Any]:
        accuracy_score = 3.0
        tone_score = 3.0
        fairness_score = 3.0
        consistency_score = 3.0
        hallucination_flag = 0
        bias_flag = 0

        return {
            "judge_id": judge_config["judge_id"],
            "model": judge_config["model"],
            "prompt_style": judge_config["label"],
            "verdict": "partial",
            "partial_correct": 1,
            "accuracy_score": accuracy_score,
            "tone_score": tone_score,
            "fairness_score": fairness_score,
            "consistency_score": consistency_score,
            "hallucination_flag": hallucination_flag,
            "bias_flag": bias_flag,
            "overall_score": self._combine_scores(
                accuracy_score,
                tone_score,
                fairness_score,
                consistency_score,
                hallucination_flag,
                bias_flag,
            ),
            "reasoning": f"Fallback neutral score because judge call failed: {error}",
            "judge_tokens_used": 0,
            "estimated_cost_usd": 0.0,
            "error": error,
        }

    def _normalize_judge_result(self, raw_result: Dict[str, Any], judge_config: Dict[str, Any], tokens_used: int) -> Dict[str, Any]:
        verdict = self._coerce_verdict(raw_result.get("verdict"))
        partial_correct = self._coerce_flag(raw_result.get("partial_correct", verdict == "partial"))
        accuracy_score = self._coerce_score(raw_result.get("accuracy_score"))
        tone_score = self._coerce_score(raw_result.get("tone_score"))
        fairness_score = self._coerce_score(raw_result.get("fairness_score"))
        consistency_score = self._coerce_score(raw_result.get("consistency_score"))
        hallucination_flag = self._coerce_flag(raw_result.get("hallucination_flag"))
        bias_flag = self._coerce_flag(raw_result.get("bias_flag"))
        reasoning = str(raw_result.get("reasoning", "")).strip()

        return {
            "judge_id": judge_config["judge_id"],
            "model": judge_config["model"],
            "prompt_style": judge_config["label"],
            "verdict": verdict,
            "partial_correct": partial_correct,
            "accuracy_score": accuracy_score,
            "tone_score": tone_score,
            "fairness_score": fairness_score,
            "consistency_score": consistency_score,
            "hallucination_flag": hallucination_flag,
            "bias_flag": bias_flag,
            "overall_score": self._combine_scores(
                accuracy_score,
                tone_score,
                fairness_score,
                consistency_score,
                hallucination_flag,
                bias_flag,
            ),
            "reasoning": reasoning,
            "judge_tokens_used": tokens_used,
            "estimated_cost_usd": round((tokens_used / 1000) * self.cost_per_1k_tokens, 6),
        }

    async def _evaluate_with_prompt(
        self,
        judge_config: Dict[str, Any],
        question: str,
        answer: str,
        ground_truth: str,
    ) -> Dict[str, Any]:
        prompt = self._build_prompt(question, answer, ground_truth)

        try:
            response = await self.openai_client.chat.completions.create(
                model=judge_config["model"],
                messages=[
                    {"role": "system", "content": judge_config["system_prompt"]},
                    {"role": "user", "content": prompt},
                ],
                temperature=judge_config["temperature"],
                max_tokens=400,
            )

            content = (response.choices[0].message.content or "").strip()
            parsed = self._extract_json(content)
            usage = getattr(response, "usage", None)
            total_tokens = int(getattr(usage, "total_tokens", 0) or 0)
            return self._normalize_judge_result(parsed, judge_config, total_tokens)
        except Exception as exc:
            print(f"Judge error ({judge_config['judge_id']}): {exc}")
            return self._default_result(judge_config, str(exc))

    async def _evaluate_with_gpt(self, question: str, answer: str, ground_truth: str) -> Dict[str, Any]:
        return await self._evaluate_with_prompt(self.judge_configs[0], question, answer, ground_truth)

    async def _evaluate_with_claude_simulation(self, question: str, answer: str, ground_truth: str) -> Dict[str, Any]:
        return await self._evaluate_with_prompt(self.judge_configs[1], question, answer, ground_truth)

    def _calculate_agreement_rate(self, score_a: float, score_b: float) -> float:
        diff = abs(score_a - score_b)
        if diff <= 0.25:
            return 1.0
        if diff <= 0.5:
            return 0.9
        if diff <= 1.0:
            return 0.75
        if diff <= 1.5:
            return 0.55
        return 0.35

    def _resolve_conflict(self, score_a: float, score_b: float) -> Tuple[float, str]:
        diff = abs(score_a - score_b)
        if diff > self.large_conflict_threshold:
            return min(score_a, score_b), "lower_score_due_to_large_gap"
        return (score_a + score_b) / 2, "average"

    def _merge_verdict(self, verdict_a: str, verdict_b: str, final_score: float) -> str:
        if verdict_a == verdict_b:
            return verdict_a
        if final_score >= 4.0:
            return "correct"
        if final_score < 2.5:
            return "incorrect"
        return "partial"

    def _build_review_reasons(self, judge_a_result: Dict[str, Any], judge_b_result: Dict[str, Any], agreement_rate: float) -> List[str]:
        reasons: List[str] = []
        if agreement_rate < 0.75:
            reasons.append("low_judge_agreement")
        if judge_a_result.get("hallucination_flag") or judge_b_result.get("hallucination_flag"):
            reasons.append("hallucination_flagged")
        if judge_a_result.get("bias_flag") or judge_b_result.get("bias_flag"):
            reasons.append("bias_flagged")
        if judge_a_result.get("verdict") != judge_b_result.get("verdict"):
            reasons.append("verdict_mismatch")
        if judge_a_result.get("error") or judge_b_result.get("error"):
            reasons.append("judge_error")
        return reasons

    async def evaluate_multi_judge(self, question: str, answer: str, ground_truth: str) -> Dict[str, Any]:
        judge_a_result, judge_b_result = await asyncio.gather(
            self._evaluate_with_gpt(question, answer, ground_truth),
            self._evaluate_with_claude_simulation(question, answer, ground_truth),
        )

        score_a = judge_a_result.get("overall_score", 3.0)
        score_b = judge_b_result.get("overall_score", 3.0)
        agreement_rate = self._calculate_agreement_rate(score_a, score_b)
        final_score, conflict_resolution = self._resolve_conflict(score_a, score_b)

        merged_dimensions = {
            "accuracy": round((judge_a_result["accuracy_score"] + judge_b_result["accuracy_score"]) / 2, 2),
            "tone": round((judge_a_result["tone_score"] + judge_b_result["tone_score"]) / 2, 2),
            "fairness": round((judge_a_result["fairness_score"] + judge_b_result["fairness_score"]) / 2, 2),
            "consistency": round((judge_a_result["consistency_score"] + judge_b_result["consistency_score"]) / 2, 2),
        }

        merged_flags = {
            "hallucination": max(judge_a_result["hallucination_flag"], judge_b_result["hallucination_flag"]),
            "bias": max(judge_a_result["bias_flag"], judge_b_result["bias_flag"]),
            "partial_correct": max(judge_a_result["partial_correct"], judge_b_result["partial_correct"]),
        }

        merged_verdict = self._merge_verdict(judge_a_result["verdict"], judge_b_result["verdict"], final_score)
        review_reasons = self._build_review_reasons(judge_a_result, judge_b_result, agreement_rate)
        total_judge_tokens = int(judge_a_result.get("judge_tokens_used", 0) + judge_b_result.get("judge_tokens_used", 0))
        total_judge_cost = round(
            judge_a_result.get("estimated_cost_usd", 0.0) + judge_b_result.get("estimated_cost_usd", 0.0),
            6,
        )
        user_satisfaction_score = round((merged_dimensions["tone"] * 0.6) + (merged_dimensions["fairness"] * 0.4), 2)

        return {
            "final_score": round(final_score, 2),
            "agreement_rate": round(agreement_rate, 2),
            "verdict": merged_verdict,
            "manual_review_recommended": bool(review_reasons),
            "review_reasons": review_reasons,
            "individual_scores": {
                judge_a_result["judge_id"]: round(score_a, 2),
                judge_b_result["judge_id"]: round(score_b, 2),
            },
            "judge_models": {
                judge_a_result["judge_id"]: judge_a_result["model"],
                judge_b_result["judge_id"]: judge_b_result["model"],
            },
            "dimension_scores": {
                judge_a_result["judge_id"]: {
                    "accuracy": judge_a_result["accuracy_score"],
                    "tone": judge_a_result["tone_score"],
                    "fairness": judge_a_result["fairness_score"],
                    "consistency": judge_a_result["consistency_score"],
                },
                judge_b_result["judge_id"]: {
                    "accuracy": judge_b_result["accuracy_score"],
                    "tone": judge_b_result["tone_score"],
                    "fairness": judge_b_result["fairness_score"],
                    "consistency": judge_b_result["consistency_score"],
                },
                "merged": merged_dimensions,
            },
            "flags": {
                judge_a_result["judge_id"]: {
                    "hallucination": judge_a_result["hallucination_flag"],
                    "bias": judge_a_result["bias_flag"],
                    "partial_correct": judge_a_result["partial_correct"],
                },
                judge_b_result["judge_id"]: {
                    "hallucination": judge_b_result["hallucination_flag"],
                    "bias": judge_b_result["bias_flag"],
                    "partial_correct": judge_b_result["partial_correct"],
                },
                "merged": merged_flags,
            },
            "verdicts": {
                judge_a_result["judge_id"]: judge_a_result["verdict"],
                judge_b_result["judge_id"]: judge_b_result["verdict"],
                "merged": merged_verdict,
            },
            "reasoning": {
                judge_a_result["judge_id"]: judge_a_result.get("reasoning", ""),
                judge_b_result["judge_id"]: judge_b_result.get("reasoning", ""),
            },
            "consistency_score": merged_dimensions["consistency"],
            "fairness_score": merged_dimensions["fairness"],
            "user_satisfaction_score": user_satisfaction_score,
            "judge_tokens_used": total_judge_tokens,
            "estimated_judge_cost_usd": total_judge_cost,
            "conflict_resolution": conflict_resolution,
            "judges": [judge_a_result, judge_b_result],
        }

    async def check_position_bias(self, response_a: str, response_b: str, question: str = "Compare") -> Dict[str, Any]:
        result_ab = await self.evaluate_multi_judge(question, response_a, response_b)
        result_ba = await self.evaluate_multi_judge(question, response_b, response_a)

        bias_score = abs(result_ab["final_score"] - result_ba["final_score"])

        return {
            "has_position_bias": bias_score > 0.5,
            "bias_magnitude": round(bias_score, 2),
            "original_score": result_ab["final_score"],
            "reversed_score": result_ba["final_score"],
        }
