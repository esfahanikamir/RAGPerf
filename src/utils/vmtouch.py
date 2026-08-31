import subprocess
import re

def check_vmtouch_residency(path):
    """
    Runs `vmtouch -v <path>` and returns a dict with residency stats.
    Returns None if vmtouch isn't available or the call fails.
    """
    try:
        result = subprocess.run(
            ["vmtouch", "-v", path],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except FileNotFoundError:
        print("[warn] vmtouch not found on this system")
        return None
    except subprocess.TimeoutExpired:
        print(f"[warn] vmtouch timed out on {path}")
        return None

    output = result.stdout

    return(output)