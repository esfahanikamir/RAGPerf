#!/usr/bin/env python3

from pathlib import Path
import json

from lance.file import LanceFileReader


DATASET = Path(
    "/upb/users/a/amirk/scratch/UPB_phd/RAG/RAGPerf/"
    "RAGPerf_dbs/lance_pdfimage/"
    "vidore_v3_physics_2.lance"
)

INDEX_UUID = "6ce55ff3-5ee1-46e5-8194-e27e59c477e6"

INDEX_PATH = (
    DATASET
    / "_indices"
    / INDEX_UUID
    / "index.idx"
)

AUX_PATH = (
    DATASET
    / "_indices"
    / INDEX_UUID
    / "auxiliary.idx"
)


def main():

    print("=" * 80)
    print("INSPECT 10TH ELEMENT OF EACH HNSW CLUSTER")
    print("=" * 80)

    # --------------------------------------------------------
    # Open index
    # --------------------------------------------------------

    index_reader = LanceFileReader(
        str(INDEX_PATH)
    )

    index_meta = index_reader.metadata()

    print("\nindex.idx rows:")
    print(index_meta.num_rows)

    # --------------------------------------------------------
    # HNSW metadata
    # --------------------------------------------------------

    raw = (
        index_meta
        .schema
        .metadata[b"lance:hnsw"]
    )

    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")

    graphs = json.loads(raw)

    print(
        "Number of HNSW graphs:",
        len(graphs)
    )

    if len(graphs) != 256:
        raise RuntimeError(
            f"Expected 256 graphs, "
            f"got {len(graphs)}"
        )

    # --------------------------------------------------------
    # Print metadata summary
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("HNSW GRAPH METADATA")
    print("=" * 80)

    print(
        f"{'cluster':>8} "
        f"{'entry':>8} "
        f"{'level0':>12} "
        f"{'total':>12}"
    )

    for i, graph in enumerate(graphs):

        if isinstance(graph, str):
            graph = json.loads(graph)

        offsets = graph[
            "level_offsets"
        ]

        print(
            f"{i:8d} "
            f"{graph['entry_point']:8d} "
            f"{offsets[1]:12d} "
            f"{offsets[-1]:12d}"
        )

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # We are NOT going to infer graph physical positions
    # from level_offsets[-1].
    #
    # Instead, first inspect the actual index rows and
    # determine how the graphs are laid out.
    #
    # Read the whole index table.
    #
    # 1.8M rows is manageable if your machine has enough RAM,
    # but we can reduce this later.
    # --------------------------------------------------------

    print("\nReading index.idx into Arrow table...")

    index_table = (
        index_reader
        .read_all(
            batch_size=65536
        )
        .to_table()
    )

    print(
        "Actual index rows read:",
        index_table.num_rows
    )

    # --------------------------------------------------------
    # Inspect vector IDs
    # --------------------------------------------------------

    vector_ids = (
        index_table[
            "__vector_id"
        ]
    )

    print("\n" + "=" * 80)
    print("FIRST 100 INDEX ROWS")
    print("=" * 80)

    for i in range(
        min(100, index_table.num_rows)
    ):

        print(
            f"index row {i:6d} "
            f"__vector_id="
            f"{vector_ids[i].as_py()}"
        )

    # --------------------------------------------------------
    # Find the beginning of each graph by looking for
    # __vector_id == 0.
    #
    # Since each HNSW graph uses local vector IDs,
    # every graph should contain a vector_id 0.
    #
    # This is much safer than assuming that
    # level_offsets[-1] is the physical graph size.
    # --------------------------------------------------------

    print("\nSearching for __vector_id == 0...")

    graph_starts = []

    for i in range(
        index_table.num_rows
    ):

        value = (
            vector_ids[i].as_py()
        )

        if value == 0:

            graph_starts.append(i)

    print(
        "Number of __vector_id=0 occurrences:",
        len(graph_starts)
    )

    print(
        "First occurrences:",
        graph_starts[:30]
    )

    # --------------------------------------------------------
    # We expect one beginning for each HNSW graph.
    # --------------------------------------------------------

    if len(graph_starts) != 256:

        print(
            "\nWARNING:"
        )

        print(
            "Expected 256 occurrences of "
            "__vector_id=0."
        )

        print(
            "Found:",
            len(graph_starts)
        )

        print(
            "\nThis means that __vector_id=0 is "
            "not sufficient to identify graph starts."
        )

        print(
            "We will NOT guess the graph boundaries."
        )

        return

    # --------------------------------------------------------
    # Now take the 10th physical row of every graph.
    #
    # 1st = position 0
    # 10th = position 9
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("10TH ELEMENT OF EACH GRAPH")
    print("=" * 80)

    tenth_rows = []

    for cluster in range(256):

        graph_start = graph_starts[
            cluster
        ]

        physical_row = (
            graph_start + 9
        )

        if physical_row >= index_table.num_rows:

            raise RuntimeError(
                f"Cluster {cluster}: "
                "10th row exceeds index"
            )

        vector_id = (
            index_table[
                "__vector_id"
            ][physical_row]
            .as_py()
        )

        neighbors = (
            index_table[
                "__neighbors"
            ][physical_row]
            .as_py()
        )

        distances = (
            index_table[
                "_distance"
            ][physical_row]
            .as_py()
        )

        tenth_rows.append(
            {
                "cluster": cluster,

                "index_row":
                    physical_row,

                "vector_id":
                    vector_id,

                "num_neighbors":
                    len(neighbors),

                "neighbors":
                    neighbors,

                "distances":
                    distances,
            }
        )

        print(
            f"\ncluster {cluster:3d}"
        )

        print(
            f"  index row:       {physical_row}"
        )

        print(
            f"  __vector_id:     {vector_id}"
        )

        print(
            f"  num neighbors:   {len(neighbors)}"
        )

        print(
            f"  neighbors:       {neighbors}"
        )

        print(
            f"  distances:       {distances}"
        )

    # --------------------------------------------------------
    # Auxiliary table
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("AUXILIARY INDEX")
    print("=" * 80)

    aux_reader = LanceFileReader(
        str(AUX_PATH)
    )

    aux_meta = aux_reader.metadata()

    print(
        "auxiliary.idx rows:",
        aux_meta.num_rows
    )

    # --------------------------------------------------------
    # Read first 1000 auxiliary rows just for inspection.
    # --------------------------------------------------------

    aux_table = (
        aux_reader
        .read_all(
            batch_size=1000
        )
        .to_table()
    )

    print(
        "Read auxiliary rows:",
        aux_table.num_rows
    )

    print("\nFirst 20 auxiliary rows:")

    for i in range(
        min(20, aux_table.num_rows)
    ):

        rowid = (
            aux_table[
                "_rowid"
            ][i]
            .as_py()
        )

        sq = (
            aux_table[
                "__sq_code"
            ][i]
            .as_py()
        )

        print(
            f"aux row {i:5d}: "
            f"_rowid={rowid:10d} "
            f"SQ={sq[:10]}"
        )

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # We deliberately stop here.
    #
    # We have NOT yet assumed:
    #
    #     HNSW vector_id -> auxiliary row
    #
    # because that mapping needs to be demonstrated from
    # the actual Lance index layout.
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)

    print(
        "Successfully extracted the 10th HNSW "
        "element from each graph."
    )

    print(
        "No unverified HNSW -> auxiliary mapping "
        "was assumed."
    )


if __name__ == "__main__":
    main()