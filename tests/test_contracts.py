from __future__ import annotations

import unittest

from aba_deep_learning import (
    ContractError,
    FEATURE_VECTOR_LENGTH,
    validate_action_mask,
    validate_action_request,
    validate_observation,
    validate_stream_records,
)


def observation(step_index: int = 0) -> dict:
    return {
        "schema_version": "1.0.0",
        "episode_id": "fixture-episode",
        "step_index": step_index,
        "timestamp_ms": 1000 + step_index * 67,
        "dt_ms": 0 if step_index == 0 else 67,
        "self_state": {},
        "target_state": {},
        "relative": {},
        "combat": {},
        "cooldowns": {},
        "network": {},
        "history": {},
        "feature_vector": [0.0] * FEATURE_VECTOR_LENGTH,
    }


def action_mask() -> dict:
    return {
        "schema_version": "1.0.0",
        "commands": {
            "none": True,
            "m1": True,
            "block_start": True,
            "block_stop": False,
            "dodge": True,
            "jump": True,
            "move": True,
            "transform": False,
            "reset_spacing": True,
        },
        "move_slots": [True, True, True, True],
        "reasons": {},
        "confidence": 0.9,
    }


def action() -> dict:
    return {
        "schema_version": "1.0.0",
        "intent": "approach",
        "move_x": 0.0,
        "move_z": 1.0,
        "sprint": 1.0,
        "facing_mode": "face_target",
        "command": "none",
        "move_slot": None,
        "delay_ms": 0,
        "hold_ms": 0,
        "confidence": 0.8,
        "policy_id": "fixture-policy",
        "model_version": "fixture-model",
    }


def step(step_index: int = 0) -> dict:
    return {
        "observation": observation(step_index),
        "action_mask": action_mask(),
        "action_request": action(),
        "executor_result": {"status": "unobserved"},
        "confirmations": [],
        "rewards": {
            "damage_dealt": 0.0,
            "damage_received": 0.0,
            "invalid_request": 0.0,
        },
        "raw_events": [],
    }


class ContractTests(unittest.TestCase):
    def test_observation_requires_exactly_64_features(self):
        value = observation()
        validate_observation(value)
        value["feature_vector"].pop()
        with self.assertRaises(ContractError):
            validate_observation(value)

    def test_move_request_requires_slot(self):
        value = action()
        value["command"] = "move"
        with self.assertRaises(ContractError):
            validate_action_request(value)
        value["move_slot"] = 2
        validate_action_request(value)

    def test_mask_requires_all_commands(self):
        value = action_mask()
        del value["commands"]["dodge"]
        with self.assertRaises(ContractError):
            validate_action_mask(value)

    def test_jsonl_record_order_and_episode_identity(self):
        records = [
            {
                "record_type": "header",
                "schema_version": "1.0.0",
                "header": {
                    "episode_id": "fixture-episode",
                    "stream_format": "aba_episode_jsonl_v1",
                },
            },
            {"record_type": "step", "schema_version": "1.0.0", "step": step(0)},
            {
                "record_type": "terminal",
                "schema_version": "1.0.0",
                "terminal": {"reason": "fixture", "duration_ms": 0},
            },
        ]
        rebuilt = validate_stream_records(records)
        self.assertEqual(len(rebuilt["steps"]), 1)

    def test_luau_bundle_is_observation_only(self):
        source = open("runtime_luau/collector_bundle.client.luau", encoding="utf-8").read()
        for forbidden in (":FireServer(", "VirtualInputManager", "mouse1click", "keypress("):
            self.assertNotIn(forbidden, source)
        self.assertIn("ABA_NEURAL_COLLECTOR", source)
        self.assertIn("FEATURE_COUNT = 64", source)


if __name__ == "__main__":
    unittest.main()
