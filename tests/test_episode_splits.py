from __future__ import annotations

import unittest

from aba_deep_learning.episode_splits import (
    EpisodeSplitConfig,
    EpisodeSplitError,
    plan_episode_splits,
    validate_episode_split_plan,
)


def episode(
    episode_id: str,
    *,
    job_id: str | None = None,
    schema_id: str | None = None,
    width: int = 64,
    steps: int = 10,
) -> dict:
    header = {
        "episode_id": episode_id,
        "stream_format": "aba_episode_jsonl_v1",
        "job_id": job_id or f"job-{episode_id}",
    }
    if schema_id:
        header["feature_schema_id"] = schema_id
    rows = [
        {
            "observation": {
                "feature_vector": [0.0] * width,
            }
        }
        for _ in range(steps)
    ]
    return {
        "header": header,
        "steps": rows,
        "terminal": {"duration_ms": steps * 67},
    }


class EpisodeSplitTests(unittest.TestCase):
    def test_three_independent_groups_fill_all_splits(self) -> None:
        plan = plan_episode_splits([
            episode("a"),
            episode("b"),
            episode("c"),
        ])
        partition = plan["schema_partitions"]["observation_v1_64"]
        self.assertTrue(partition["complete"])
        self.assertEqual(
            set(partition["assignments"].values()),
            {"train", "validation", "test"},
        )
        validate_episode_split_plan(plan)

    def test_duplicate_job_collapses_independence(self) -> None:
        plan = plan_episode_splits([
            episode("a", job_id="same-match"),
            episode("b", job_id="same-match"),
            episode("c", job_id="other-match"),
        ])
        partition = plan["schema_partitions"]["observation_v1_64"]
        self.assertEqual(partition["independent_groups"], 2)
        self.assertFalse(partition["complete"])
        self.assertEqual(
            partition["assignments"]["a"],
            partition["assignments"]["b"],
        )
        self.assertIn(
            "non_independent_episode_groups_collapsed",
            partition["warnings"],
        )
        validate_episode_split_plan(plan)

    def test_require_complete_rejects_two_groups(self) -> None:
        with self.assertRaises(EpisodeSplitError):
            plan_episode_splits(
                [episode("a"), episode("b")],
                EpisodeSplitConfig(require_complete_splits=True),
            )

    def test_plan_is_deterministic(self) -> None:
        values = [episode(str(index)) for index in range(7)]
        self.assertEqual(
            plan_episode_splits(values),
            plan_episode_splits(reversed(values)),
        )

    def test_feature_schemas_are_partitioned(self) -> None:
        plan = plan_episode_splits([
            episode("legacy"),
            episode(
                "ranked",
                schema_id="ranked_explicit_v3_72",
                width=72,
            ),
        ])
        self.assertEqual(
            set(plan["schema_partitions"]),
            {"observation_v1_64", "ranked_explicit_v3_72"},
        )

    def test_larger_plan_preserves_every_episode_once(self) -> None:
        values = [episode(str(index), steps=index + 5) for index in range(13)]
        plan = plan_episode_splits(values)
        partition = plan["schema_partitions"]["observation_v1_64"]
        assignments = partition["assignments"]
        self.assertEqual(set(assignments), {str(index) for index in range(13)})
        self.assertEqual(sum(partition["episode_counts_by_split"].values()), 13)
        self.assertEqual(
            sum(partition["steps_by_split"].values()),
            sum(index + 5 for index in range(13)),
        )
        validate_episode_split_plan(plan)


if __name__ == "__main__":
    unittest.main()
