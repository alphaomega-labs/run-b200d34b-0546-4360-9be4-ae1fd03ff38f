from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run QRC hybrid validation simulations")
    parser.add_argument("--workspace-root", default=".", help="Workspace root path")
    parser.add_argument("--iteration-index", type=int, default=None, help="Iteration index for iter_<n> output folders")
    parser.add_argument(
        "--result-json",
        default=None,
        help="Path to write compact run result JSON (relative to workspace root)",
    )
    args = parser.parse_args()

    workspace_root = Path(args.workspace_root).resolve()
    src_path = workspace_root / "experiments" / "qrc_hybrid_validation" / "src"
    sys.path.insert(0, str(src_path))

    from qrc_validation import run_pipeline

    result = run_pipeline(str(workspace_root), iteration_index=args.iteration_index)

    if args.result_json:
        out_path = workspace_root / args.result_json
    elif args.iteration_index is None:
        out_path = workspace_root / "experiments" / "qrc_hybrid_validation" / "outputs" / "run_result.json"
    else:
        out_path = (
            workspace_root
            / "experiments"
            / "qrc_hybrid_validation"
            / f"iter_{args.iteration_index}"
            / "outputs"
            / "run_result.json"
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
