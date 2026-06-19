import os
import time
import psutil

class StorageTransferMonitor:
    def __init__(self, label="Operation"):
        self.label = label
        self.process = psutil.Process(os.getpid())
        self.start_io = None
        self.start_time = None

    def __enter__(self):
        import gc; gc.collect()
        self.start_io = self.process.io_counters()
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = time.perf_counter()
        end_io = self.process.io_counters()
        elapsed = end_time - self.start_time
        
        bytes_phys = end_io.read_bytes - self.start_io.read_bytes
        mb_phys = bytes_phys / (1024 ** 2)
        
        chars_start = getattr(self.start_io, 'read_chars', self.start_io.read_bytes)
        chars_end = getattr(end_io, 'read_chars', end_io.read_bytes)
        bytes_logical = chars_end - chars_start
        mb_logical = bytes_logical / (1024 ** 2)
        
        print(f"\n[ MONITOR: {self.label} ]")
        print(f"  • Logical Transfer (App Level): {mb_logical:.4f} MB")
        print(f"  • Physical Transfer (From Disk): {mb_phys:.4f} MB")
        print(f"  • Duration:                     {elapsed:.4f} seconds")



def run_direct_io_test(num_reps = 2):
    # 1. Direct I/O must align with disk sector blocks (commonly 4096 bytes)
    block_size = 4096
    total_blocks = 2560  # 4096 * 2560 = Exactly 10.0 MB
    # dummy_data = b"X" * (block_size * total_blocks)
    
    # temp_filename = os.path.join(os.getcwd(), "direct_io_verify_10mb.bin")
    
    temp_filename = "2MB_sample1.txt"
    for i in range(num_reps):
        print(f"ROUND{i}: Executing Direct-to-Disk I/O profile test...")

        try:
            # 2. Use os.O_DIRECT to force a physical disk read payload
            with StorageTransferMonitor("Testing Raw Physical Storage Read"):
                # O_DIRECT forces the OS to skip RAM caching layers completely
                fd = os.open(temp_filename, os.O_RDONLY | os.O_DIRECT)
                try:
                    _ = os.read(fd, block_size * total_blocks)
                finally:
                    os.close(fd)
                    
        finally:
            if os.path.exists(temp_filename):
                # os.remove(temp_filename)
                print("\nVerification file safely cleaned up.")

def evict_file_from_ram(filepath):
    """Tells the OS kernel to immediately drop this file from the global Page Cache."""
    if os.path.exists(filepath):
        fd = os.open(filepath, os.O_RDONLY)
        try:
            # POSIX_FADV_DONTNEED hints to the kernel: 'Evict this from RAM now'
            os.posix_fadvise(fd, 0, os.path.getsize(filepath), os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)


def normal_io_test(evict: bool, num_reps = 2):
    # 1. Direct I/O must align with disk sector blocks (commonly 4096 bytes)
    block_size = 4096
    total_blocks = 2560  # 4096 * 2560 = Exactly 10.0 MB
    # dummy_data = b"X" * (block_size * total_blocks)
    
    # temp_filename = os.path.join(os.getcwd(), "direct_io_verify_10mb.bin")

    temp_filename = "2MB_sample1.txt"
    if evict:
        evict_file_from_ram(temp_filename)
    for i in range(num_reps):
        print(f"ROUND{i}: Executing Direct-to-Disk I/O profile test...")
        with StorageTransferMonitor("Testing Raw Physical Storage Read"):
                with open(temp_filename, "r") as f:
                    txt = f.read()


import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # normal / no-cache
    parser.add_argument("--type", required=True)
    parser.add_argument("--evict", action="store_true" )
    args = parser.parse_args()
    if args.type == "normal":
        normal_io_test(evict = args.evict)
    elif args.type == "no_cache":
        run_direct_io_test()