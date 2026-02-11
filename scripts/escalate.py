#!/usr/bin/env python3
"""
Escalate helper: filter results.jsonl to emit only items that should be escalated.

Usage:
    python scripts/escalate.py results.jsonl
    python scripts/escalate.py results.jsonl --threshold 0.05

If --threshold is provided, escalates items where instability_score > threshold.
If --threshold is not provided, escalates items where passed_threshold is false.
"""

import argparse
import json
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Filter results.jsonl to escalate items to GPT-4 or external judge"
    )
    parser.add_argument(
        "results_file",
        help="Path to results JSONL file (output from logoslabs.cli)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        help="Optional threshold override: escalate if instability_score > threshold",
    )
    
    args = parser.parse_args()
    
    try:
        with open(args.results_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                    
                try:
                    item = json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"Warning: Skipping invalid JSON line: {e}", file=sys.stderr)
                    continue
                
                # Determine if item should be escalated
                should_escalate = False
                
                if args.threshold is not None:
                    # Use custom threshold
                    score = item.get("instability_score")
                    if score is not None:
                        should_escalate = score > args.threshold
                    else:
                        print(f"Warning: Item missing 'instability_score': {item.get('id', 'unknown')}", file=sys.stderr)
                else:
                    # Use passed_threshold field (escalate when false)
                    passed = item.get("passed_threshold")
                    if passed is not None:
                        should_escalate = not passed
                    else:
                        print(f"Warning: Item missing 'passed_threshold': {item.get('id', 'unknown')}", file=sys.stderr)
                
                # Emit escalated items to stdout as JSONL
                if should_escalate:
                    print(json.dumps(item))
                    
    except FileNotFoundError:
        print(f"Error: File not found: {args.results_file}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
