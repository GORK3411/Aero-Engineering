import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).parent
PROCESSED_DIR = ROOT / "data" / "processed"
RAW_DIR       = ROOT / "data" / "raw"
RECON_DIR     = ROOT / "data" / "reconstructions"


# ── Helpers ──────────────────────────────────────────────────────────────────

def get_ranges(source: str):
    """Return list of date-range folder names for a given source."""
    if source == "cleaned":
        base = PROCESSED_DIR
    elif source == "reconstructed":
        base = RECON_DIR
    else:
        print(f"[error] Unknown source '{source}'. Use 'cleaned' or 'reconstructed'.")
        sys.exit(1)

    if not base.exists():
        print(f"[error] Directory not found: {base}")
        sys.exit(1)

    return sorted([d.name for d in base.iterdir() if d.is_dir()])


def get_flights(source: str, range_name: str):
    """Return list of flight folder names inside a date-range."""
    if source == "cleaned":
        base = PROCESSED_DIR
    else:
        base = RECON_DIR

    range_path = base / range_name
    if not range_path.exists():
        print(f"[error] Range '{range_name}' not found in {base}")
        sys.exit(1)

    return sorted([d.name for d in range_path.iterdir() if d.is_dir()])


# ── Commands ──────────────────────────────────────────────────────────────────

def cmd_list(args):
    if args.range:
        flights = get_flights(args.source, args.range)
        print(f"\nFlights in [{args.source}] → {args.range}:")
        for f in flights:
            print(f"  {f}")
        print(f"\n  Total: {len(flights)} flight(s)")
    else:
        ranges = get_ranges(args.source)
        print(f"\nAvailable date ranges in [{args.source}]:")
        for r in ranges:
            print(f"  {r}")
        print(f"\n  Total: {len(ranges)} range(s)")


def cmd_check_missing(args):
    if not RAW_DIR.exists():
        print(f"[error] Raw directory not found: {RAW_DIR}")
        sys.exit(1)
    if not PROCESSED_DIR.exists():
        print(f"[error] Processed directory not found: {PROCESSED_DIR}")
        sys.exit(1)

    raw_ranges = {d.name for d in RAW_DIR.iterdir() if d.is_dir()}
    proc_ranges = {d.name for d in PROCESSED_DIR.iterdir() if d.is_dir()}

    missing_ranges = raw_ranges - proc_ranges
    if missing_ranges:
        print(f"\n[!] Date ranges in raw but missing in processed:")
        for r in sorted(missing_ranges):
            print(f"  {r}")

    any_missing = False
    for range_name in sorted(raw_ranges & proc_ranges):
        raw_flights  = {d.name for d in (RAW_DIR / range_name).iterdir() if d.is_dir()}
        proc_flights = {d.name for d in (PROCESSED_DIR / range_name).iterdir() if d.is_dir()}
        missing = raw_flights - proc_flights
        if missing:
            any_missing = True
            print(f"\n[!] Missing in processed/{range_name}:")
            for f in sorted(missing):
                print(f"    {f}")

    if not missing_ranges and not any_missing:
        print("\n[ok] All raw flights are present in processed.")


def cmd_reconstruct(args):
    # Validate inputs
    flight_path = PROCESSED_DIR / args.range / args.flight
    if not flight_path.exists():
        print(f"[error] Flight not found: {flight_path}")
        print(f"  Tip: run 'python main.py list --source cleaned --range {args.range}' to see available flights.")
        sys.exit(1)

    import subprocess
    script = ROOT / "src" / "reconstruction.py"
    result = subprocess.run(
        [sys.executable, str(script), args.range, args.flight],
        cwd=ROOT
    )
    sys.exit(result.returncode)


def cmd_display(args):
    # Validate inputs
    recon_path = RECON_DIR / args.range / args.flight / "full_reconstruction.parquet"
    if not recon_path.exists():
        print(f"[error] Reconstruction not found for this flight.")
        print(f"  Tip: run 'python main.py reconstruct --range {args.range} --flight {args.flight}' first.")
        sys.exit(1)

    import subprocess
    script = ROOT / "src" / "display.py"
    result = subprocess.run(
        [sys.executable, str(script), args.range, args.flight],
        cwd=ROOT
    )
    sys.exit(result.returncode)


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="ADS-B Trajectory Fusion — CLI"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # list
    p_list = sub.add_parser("list", help="List date ranges or flights")
    p_list.add_argument("--source", choices=["cleaned", "reconstructed"], default="cleaned",
                        help="Which data source to inspect (default: cleaned)")
    p_list.add_argument("--range", type=str, default=None,
                        help="If provided, list flights inside this date range")

    # check-missing
    sub.add_parser("check-missing", help="Check for flights missing from processed vs raw")

    # reconstruct
    p_recon = sub.add_parser("reconstruct", help="Reconstruct a flight")
    p_recon.add_argument("--range",  type=str, required=True, help="Date range folder name")
    p_recon.add_argument("--flight", type=str, required=True, help="Flight folder name")

    # display
    p_disp = sub.add_parser("display", help="Display a reconstructed flight on a map")
    p_disp.add_argument("--range",  type=str, required=True, help="Date range folder name")
    p_disp.add_argument("--flight", type=str, required=True, help="Flight folder name")

    args = parser.parse_args()

    dispatch = {
        "list":          cmd_list,
        "check-missing": cmd_check_missing,
        "reconstruct":   cmd_reconstruct,
        "display":       cmd_display,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()