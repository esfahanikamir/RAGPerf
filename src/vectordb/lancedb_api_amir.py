
import os
import psutil
import sys
import time
from .lancedb_api import lance_client

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Remove the dots completely
from src.Amir_Helpers.storage_read_bytes_test import evict_file_from_ram



sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.reverse()



class lance_client_Amir(lance_client):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.type = "lancedb"


    def query(self, collection_name, collection_abs_path, filter_expr, output_fields=None, limit=10):
        # Force hardware memory cache drop to ensure true cold-start numbers
        # print(f"DB file to be evicted from cache -> {collection_abs_path}")
        # evict_file_from_ram(collection_abs_path)
        # proc = psutil.Process(os.getpid())
        
        # -------------------------------------------------------------
        # STEP 1: OPEN TABLE METADATA
        # -------------------------------------------------------------
        # before_open = proc.io_counters()
        t0 = time.monotonic_ns()
        
        tbl = self.client.open_table(collection_name)
        
        t1 = time.monotonic_ns()
        # after_open = proc.io_counters()

        # -------------------------------------------------------------
        # STEP 2: LAZY SEARCH INITIALIZATION
        # -------------------------------------------------------------
        if output_fields is not None:
            data_plan = tbl.search().where(filter_expr).select(output_fields).limit(limit)
        else:
            data_plan = tbl.search().where(filter_expr).limit(limit)
            
        t2 = time.monotonic_ns()
        # after_setup = proc.io_counters()

        # -------------------------------------------------------------
        # STEP 3: PURE DATABASE EXECUTION (EAGER ARROW SCAN)
        # -------------------------------------------------------------
        # This forces LanceDB's Rust layer to complete all index-scanning,
        # vector calculation, and raw physical disk storage lookups.
        arrow_table = data_plan.to_arrow()
        
        t3 = time.monotonic_ns()
        # after_db_exec = proc.io_counters()

        # -------------------------------------------------------------
        # STEP 4: PANDAS DATA DATATYPE CONVERSION (CPU/MEM BOUND)
        # -------------------------------------------------------------
        # Pure Python memory space dictionary allocation and DataFrame copying.
        query_df = arrow_table.to_pandas()
        
        t4 = time.monotonic_ns()
        # after_pandas = proc.io_counters()

        # -------------------------------------------------------------
        # DICTIONARY REPORT PACKAGING
        # -------------------------------------------------------------
        db_fetch_stats = {
            "open_table_start": t0,
            "open_table_time_ns": t1 - t0,
            "lazy_search_start": t1,
            "lazy_search_time_ns": t2 - t1,       # Near zero due to lazy execution
            "db_fetch_pure_start" : t2,
            "db_fetch_pure_time_ns": t3 - t2,     # Isolated LanceDB database execution runtime
            "pandas_start": t3,
            "pandas_time_ns": t4 - t3,            # Isolated Python dataframe creation runtime
            "pandas_end" : t4
        }

        # Clean math calculation block function
        # def calculate_diffs(start_counter, end_counter, label_prefix, stats_dict):
        #     bytes_phys = end_counter.read_bytes - start_counter.read_bytes
        #     chars_start = getattr(start_counter, 'read_chars', start_counter.read_bytes)
        #     chars_end = getattr(end_counter, 'read_chars', end_counter.read_bytes)
            
        #     stats_dict[f"{label_prefix}_mb_phy"] = bytes_phys / (1024 ** 2)
        #     stats_dict[f"{label_prefix}_mb_log"] = (chars_end - chars_start) / (1024 ** 2)

        # # Run and append metrics explicitly to the reporting dictionary
        # calculate_diffs(before_open, after_open, "open_table", db_fetch_stats)
        # calculate_diffs(after_open, after_setup, "lazy_search", db_fetch_stats)
        # calculate_diffs(after_setup, after_db_exec, "db_fetch_pure", db_fetch_stats)
        # calculate_diffs(after_db_exec, after_pandas, "pandas", db_fetch_stats)

        return (query_df, db_fetch_stats)

 
