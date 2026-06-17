#!/usr/bin/env python3

import argparse
import os
import re
import subprocess
import sys
import time


def run(cmd):
    result = subprocess.run(
        cmd,
        text=True,
        capture_output=True
    )

    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr, file=sys.stderr)
        raise RuntimeError(
            f"Command failed ({result.returncode}): {' '.join(cmd)}"
        )

    return result.stdout


def get_residency_percent(target_dir):
    output = run(["vmtouch", target_dir])

    print(output)

    # Example line:
    # Resident Pages: 792495/987089  3G/3G  80.3%
    m = re.search(
        r"Resident Pages:\s+\d+/\d+\s+\S+\s+([0-9.]+)%",
        output
    )

    if not m:
        raise RuntimeError(
            "Could not parse residency percentage from vmtouch output"
        )

    return float(m.group(1))


def evict(target_dir):
    output = run(["vmtouch", "-e", target_dir])
    print(output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dir",
        default="/local/amirk/RAGPerf/RAGPerf_dbs/lance_pdfimage",
        help="Directory whose page cache should be evicted"
    )
    parser.add_argument(
        "--max-resident",
        type=float,
        default=1.0,
        help="Maximum allowed resident percentage after eviction"
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="Number of eviction attempts"
    )

    args = parser.parse_args()

    target_dir = os.path.expanduser(args.dir)

    if not os.path.exists(target_dir):
        print(f"ERROR: Path does not exist: {target_dir}")
        sys.exit(1)

    print("\n=== BEFORE ===")
    before = get_residency_percent(target_dir)
    print(f"Parsed residency: {before:.2f}%")

    for attempt in range(1, args.retries + 1):
        print(f"\n=== EVICTION ATTEMPT {attempt} ===")

        evict(target_dir)

        time.sleep(1)

        print("\n=== AFTER ===")
        after = get_residency_percent(target_dir)
        print(f"Parsed residency: {after:.2f}%")

        if after <= args.max_resident:
            print(
                f"\nSUCCESS: Residency is "
                f"{after:.2f}% (threshold={args.max_resident:.2f}%)"
            )
            sys.exit(0)

    print(
        f"\nWARNING: Residency still "
        f"{after:.2f}% after {args.retries} attempts "
        f"(threshold={args.max_resident:.2f}%)"
    )
    sys.exit(1)


if __name__ == "__main__":
    main()