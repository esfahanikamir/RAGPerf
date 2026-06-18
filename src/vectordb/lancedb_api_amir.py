
import sys, os
import time


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.reverse()
from lancedb_api import lance_client


class lance_client_Amir(lance_client):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.type = "lancedb"


    def query(self, collection_name, filter_expr, output_fields=None, limit=10):
        # #1 opening the table -> does it load the whole collection to the mem? on the storage?
        t0 = time.monotonic_ns()
        tbl = self.client.open_table(collection_name)
        t1 = time.monotonic_ns()
        if output_fields is not None:
            # #2 retrieving the data
            data = tbl.search().where(filter_expr).select(output_fields)
            t2 = time.monotonic_ns()
            # #3 converting_to_pandas
            query = data.to_pandas()
            t3 = time.monotonic_ns()
        else:
            query = tbl.search().where(filter_expr).to_pandas()
        db_fetch_stats = {"open_table_time": t1 - t0, "retrieve_by_idx_time": t2- t1, "to_pandas_time": t3 - t2}
        return (query, db_fetch_stats) 
 
