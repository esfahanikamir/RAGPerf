import os

def evict_file_pages(filepath):
    """Ask the OS to drop cached pages for this specific file. No sudo needed."""
    fd = os.open(filepath, os.O_RDONLY)
    try:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)


def evict_directory_pages(dirpath):
    """Evict every file under dirpath -- use for a whole .lance table directory."""
    evicted, failed = 0, 0
    for root, _, files in os.walk(dirpath):
        for fname in files:
            fpath = os.path.join(root, fname)
            try:
                evict_file_pages(fpath)
                evicted += 1
            except OSError as e:
                print(f"[warn] could not evict {fpath}: {e}")
                failed += 1
    print(f"[evict] {dirpath}: {evicted} files evicted, {failed} failed")