from __future__ import annotations

import argparse
import json
from pathlib import Path

from aba_deep_learning.imitation import load_policy_checkpoint
from aba_deep_learning.shadow_mode import PredictionOnlyShadowRunner, ShadowConfig, read_jsonl


def main() -> int:
    parser = argparse.ArgumentParser(description="Prediction-only JSONL shadow mode. This command cannot send gameplay input or remotes.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--thresholds", type=Path, required=True)
    parser.add_argument("--dataset-manifest-sha256", required=True)
    parser.add_argument("--model-version", required=True)
    parser.add_argument("--follow", action="store_true")
    parser.add_argument("--enable-shadow", action="store_true", help="enable prediction-only logging; does not enable gameplay control")
    args = parser.parse_args()
    threshold_document = json.loads(args.thresholds.read_text(encoding="utf-8"))
    thresholds = threshold_document.get("thresholds", threshold_document)
    config = ShadowConfig(model_version=args.model_version, dataset_manifest_sha256=args.dataset_manifest_sha256, thresholds=thresholds, enabled=args.enable_shadow)
    model = load_policy_checkpoint(args.checkpoint)
    runner = PredictionOnlyShadowRunner(model, config, args.output_jsonl)
    count = runner.run(read_jsonl(args.input_jsonl, follow=args.follow))
    print(json.dumps({"predictions_written": count, "prediction_only": True, "input_sent": False, "remote_fired": False}))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
