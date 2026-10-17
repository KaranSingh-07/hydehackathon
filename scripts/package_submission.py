"""Competition Packaging and Verification Harness for Catan Agent.

Creates the official submission archive (`submission/my_team.zip`),
generates an exact file manifest and hash checksum,
unpacks the artifact into an isolated verification directory,
and executes official match benchmarks directly against the packaged artifact.
"""
import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Tuple

# Ensure project root is in sys.path
_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from scripts.benchmark import run_benchmark


SUBMISSION_DIR = "submission"
ZIP_NAME = "my_team.zip"


# Explicit whitelist of files required for tournament runtime
REQUIRED_FILES = [
    ("agent.py", "my_team/agent.py"),
    ("core/action_prioritizer.py", "my_team/core/action_prioritizer.py"),
    ("evaluation/evaluator.py", "my_team/evaluation/evaluator.py"),
    ("evaluation/features.py", "my_team/evaluation/features.py"),
    ("search/budget_manager.py", "my_team/search/budget_manager.py"),
    ("search/selective_search.py", "my_team/search/selective_search.py"),
]


def sha256_file(filepath: str) -> str:
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def get_git_commit() -> str:
    """Retrieve current Git commit."""
    try:
        res = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
        return res.decode().strip()
    except Exception:
        return "unknown"


def package_agent(out_dir: str = SUBMISSION_DIR) -> Tuple[str, str, int]:
    """Package validated agent into submission zip with manifest."""
    os.makedirs(out_dir, exist_ok=True)
    zip_path = os.path.join(out_dir, ZIP_NAME)
    manifest_path = os.path.join(out_dir, "submission_manifest.txt")

    git_commit = get_git_commit()
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    manifest_lines = [
        "=======================================================",
        " CATAN AGENT SUBMISSION MANIFEST",
        f" Archive: {ZIP_NAME}",
        f" Created: {timestamp}",
        f" Git Commit: {git_commit}",
        " Contract: Agent(obs, playable_actions) -> int",
        "=======================================================",
        "",
        "INCLUDED FILES & CHECKSUMS (SHA-256):",
    ]

    # Create zip archive
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zip_out:
        for archive_rel_path, src_path in REQUIRED_FILES:
            if not os.path.isfile(src_path):
                raise FileNotFoundError(f"Missing mandatory submission file: {src_path}")

            # Add file to zip
            zip_out.write(src_path, arcname=archive_rel_path)

            # Record file info in manifest
            file_hash = sha256_file(src_path)
            file_size = os.path.getsize(src_path)
            manifest_lines.append(f"  {archive_rel_path:<35} {file_size:>6} B  sha256:{file_hash[:16]}...")

    zip_size = os.path.getsize(zip_path)
    zip_hash = sha256_file(zip_path)

    manifest_lines.extend([
        "",
        "ARCHIVE SUMMARY:",
        f"  Total Files: {len(REQUIRED_FILES)}",
        f"  Archive Size: {zip_size:,} bytes ({zip_size / 1024:.2f} KB)",
        f"  Archive SHA-256: {zip_hash}",
        "  Target Limit: 100 MB (Submission is < 0.1% of limit)",
        "=======================================================",
    ])

    with open(manifest_path, "w", encoding="utf-8") as f_man:
        f_man.write("\n".join(manifest_lines) + "\n")

    return zip_path, manifest_path, zip_size


def verify_packaged_artifact(zip_path: str, num_games: int = 20) -> dict:
    """Extract zip into isolated scratch directory and run official match evaluation."""
    scratch_dir = tempfile.mkdtemp(prefix="catan_sub_verify_")
    print(f"\n=======================================================")
    print(f" Verifying Packaged Submission Artifact")
    print(f" Archive: {zip_path}")
    print(f" Isolated Sandbox: {scratch_dir}")
    print(f" Test Games: {num_games} vs Bot F")
    print(f"=======================================================\n")

    try:
        # Extract archive
        with zipfile.ZipFile(zip_path, "r") as zip_in:
            zip_in.extractall(scratch_dir)

        unpacked_agent_path = os.path.join(scratch_dir, "agent.py")
        if not os.path.isfile(unpacked_agent_path):
            raise FileNotFoundError("agent.py missing at top-level of extracted archive!")

        print(f" Successfully extracted archive. Testing unpacked entry point: {unpacked_agent_path}")

        # Run official benchmark using match.play_chunk
        results = run_benchmark(
            path_a=unpacked_agent_path,
            path_b="examples/f_agent.py",
            num_games=num_games,
            seed0=990000,
            tag="packaged_artifact_verification",
        )

        return results

    finally:
        # Cleanup temporary verification directory
        shutil.rmtree(scratch_dir, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(description="Package and verify Catan submission archive.")
    parser.add_argument("--games", type=int, default=20, help="Number of verification games against Bot F")
    args = parser.parse_args()

    zip_path, manifest_path, zip_size = package_agent()
    print(f" Packaged {zip_path} ({zip_size / 1024:.2f} KB).")
    print(f" Manifest written to {manifest_path}.")

    results = verify_packaged_artifact(zip_path, num_games=args.games)

    print("\n================ VERIFICATION RESULT ================")
    print(f" Win Rate: {results['win_rate_a']*100:.1f}% ({results['wins_a']}/{results['total_games']})")
    print(f" Red Seat: {results['win_rate_a_as_seat_0']*100:.1f}% | Blue Seat: {results['win_rate_a_as_seat_1']*100:.1f}%")
    print(f" Mean CPU / Game: {results['agent_a_mean_cpu_ms']:.1f} ms (p95 Decision: {results['agent_a_p95_decision_ms']:.2f} ms)")
    print(f" Illegal Errors: {results['agent_a_errors']} | Timeouts: {results['agent_a_timeouts']}")
    print("=====================================================\n")

    if results["agent_a_errors"] == 0 and results["agent_a_timeouts"] == 0 and results["win_rate_a"] >= 0.50:
        print(" >>> PACKAGED ARTIFACT VERIFICATION PASSED! READY FOR COMPETITION SUBMISSION <<<")
    else:
        print(" !!! PACKAGED ARTIFACT VERIFICATION FAILED CRITERIA !!!")
        sys.exit(1)



if __name__ == "__main__":
    main()
