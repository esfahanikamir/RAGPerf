import os

class CoreAllocator:
    """Temporarily confine the process to a specific core set.
    Restores the previous affinity on exit (exception-safe, nestable)."""

    def __init__(self, core_set):
        self.core_set = set(core_set)

    def __enter__(self):
        self.saved = os.sched_getaffinity(0)
        bad = self.core_set - self.saved
        if bad:
            raise RuntimeError(
                f"cores {sorted(bad)} not available (allowed: {sorted(self.saved)})")
        os.sched_setaffinity(0, self.core_set)
        return self

    def __exit__(self, exc_type, exc, tb):
        os.sched_setaffinity(0, self.saved)