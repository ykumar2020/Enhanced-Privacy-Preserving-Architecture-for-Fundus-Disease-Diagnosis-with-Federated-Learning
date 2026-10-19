from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .acquire import acquire_odir
from .config import load_config, save_resolved_config
from .experiment import run_centralized, run_federated, run_local
from .manifest import audit_manifest, build_manifest, write_manifest_artifacts


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="retina-fl", description="Auditable federated retinal classification")
    commands = parser.add_subparsers(dest="command", required=True)
    acquire = commands.add_parser("acquire-odir", help="Download and locate the real public ODIR-5K data")
    acquire.add_argument("--destination", default="data/odir-kaggle")
    acquire.add_argument("--template", default="configs/corrected_odir.yaml")
    acquire.add_argument("--config-out", default="configs/runtime_colab_odir.yaml")
    acquire.add_argument("--run-root", default="runs/colab_odir")
    acquire.add_argument("--force", action="store_true")
    prepare = commands.add_parser("prepare", help="Build patient-safe ODIR manifest")
    prepare.add_argument("--config", required=True)
    audit = commands.add_parser("audit", help="Audit an existing manifest")
    audit.add_argument("--manifest", required=True)
    train = commands.add_parser("train", help="Run a real-data baseline")
    train.add_argument("mode", choices=["centralized", "local", "federated"])
    train.add_argument("--config", required=True)
    train.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "acquire-odir":
        result = acquire_odir(args.destination, args.template, args.config_out, args.run_root, args.force)
        print(json.dumps(result, indent=2))
        return
    if args.command == "prepare":
        config = load_config(args.config)
        output = Path(config["data"]["output_dir"]).resolve()
        manifest, report = build_manifest(config)
        write_manifest_artifacts(manifest, report, output)
        save_resolved_config(config, output / "resolved_config.yaml")
        print(json.dumps(report, indent=2))
        return
    if args.command == "audit":
        frame = pd.read_csv(args.manifest, dtype={"patient_id": str, "client_id": str})
        print(json.dumps(audit_manifest(frame), indent=2))
        return
    config = load_config(args.config)
    runners = {"centralized": run_centralized, "local": run_local, "federated": run_federated}
    result = runners[args.mode](config, overwrite=args.overwrite)
    print(f"Completed real-data run: {result}")


if __name__ == "__main__":
    main()

