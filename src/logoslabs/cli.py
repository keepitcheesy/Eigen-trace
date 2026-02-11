"""
Command-line interface for LogosLabs AVP.

Provides a CLI for pre-filtering LLM outputs using LogosLoss-based analysis.
"""

import argparse
import sys
import json
from typing import Optional

from .avp import AVPProcessor, load_jsonl, save_jsonl


def main(argv: Optional[list] = None) -> int:
    """
    Main CLI entry point.
    
    Args:
        argv: Command-line arguments (None = use sys.argv)
        
    Returns:
        Exit code (0 = success)
    """
    parser = argparse.ArgumentParser(
        description="LogosLabs: Pre-filter LLM outputs using LogosLoss-based AVP",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        "input",
        help="Input JSONL file with 'prediction' and 'truth' fields",
    )
    
    parser.add_argument(
        "output",
        help="Output JSONL file with instability scores and pass/fail flags",
    )
    
    parser.add_argument(
        "--threshold",
        type=float,
        default=1.0,
        help="Instability score threshold (lower = stricter filtering)",
    )
    
    parser.add_argument(
        "--grace-coeff",
        type=float,
        default=0.5,
        help="LogosLoss spectral weight coefficient",
    )
    
    parser.add_argument(
        "--phase-weight",
        type=float,
        default=0.1,
        help="LogosLoss phase weight coefficient",
    )
    
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Maximum sequence length for encoding",
    )
    
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print summary statistics to stdout",
    )
    
    parser.add_argument(
        "--no-deterministic",
        action="store_true",
        help="Disable deterministic behavior",
    )
    
    parser.add_argument(
        "--enable-belief-streams",
        action="store_true",
        help="Enable belief streams (experimental, currently not implemented)",
    )
    
    args = parser.parse_args(argv)
    
    # Validate arguments
    if args.enable_belief_streams:
        print("Warning: Belief streams not yet implemented, using structural-only mode", file=sys.stderr)
    
    try:
        # Load input
        print(f"Loading input from {args.input}...", file=sys.stderr)
        items = load_jsonl(args.input)
        print(f"Loaded {len(items)} items", file=sys.stderr)
        
        # Initialize processor
        processor = AVPProcessor(
            threshold=args.threshold,
            grace_coeff=args.grace_coeff,
            phase_weight=args.phase_weight,
            max_length=args.max_length,
            structural_only=not args.enable_belief_streams,
            deterministic=not args.no_deterministic,
        )
        
        # Process batch
        print("Processing batch...", file=sys.stderr)
        results = processor.process_batch(items)
        
        # Save output
        print(f"Saving output to {args.output}...", file=sys.stderr)
        save_jsonl(results, args.output)
        
        # Print summary if requested
        if args.summary:
            summary = processor.get_summary(results)
            print("\n=== Summary ===", file=sys.stderr)
            print(json.dumps(summary, indent=2))
        
        print("Done!", file=sys.stderr)
        return 0
        
    except FileNotFoundError as e:
        print(f"Error: File not found - {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
