import subprocess
import os

src_path = "/local/amirk/RAGPerf/src"
baseline_config_folder = "/local/amirk/RAGPerf/config/pdfimage"

baseline_insert_config_filename = "lance_insert_pdfimage.yaml"
baseline_insert_msys_filepath = "/local/amirk/RAGPerf/config/monitor/minimal.yaml"
baseline_query_msys_filepath = "/local/amirk/RAGPerf/config/monitor/example_config.yaml"

baseline_query_config_filename = "lance_query_pdfimage_rerank.yaml"

# dataset_ratios = [0.001, 0.01, 0.1]
dataset_ratios = [0.01]
# dataset_ratios = [0.0001]


def update_config_lines(lines: list[str], dataset_ratio: float) -> list[str]:
    """Update dataset_ratio and collection_name entries."""

    for i, line in enumerate(lines):
        stripped = line.lstrip()
        indent_size = len(line) - len(stripped)

        if stripped.startswith("dataset_ratio"):
            lines[i] = (
                f"{' ' * indent_size}dataset_ratio: {dataset_ratio}\n"
            )

        elif stripped.startswith("collection_name"):
            lines[i] = (
                f"{' ' * indent_size}"
                f"collection_name: 'lance_image_test_{str(dataset_ratio).split('.')[-1]}'\n"
            )

    return lines


def gen_new_config_file(dataset_ratio: float) -> dict[str, str]:
    """
    Generates new insert/query config files, saves them,
    and returns their absolute paths.
    """

    ratio_decimal = str(dataset_ratio).split(".")[-1]

    insert_cfg_filename = (
        f"{baseline_insert_config_filename.rsplit('.', 1)[0]}_{ratio_decimal}.yaml"
    )

    query_cfg_filename = (
        f"{baseline_query_config_filename.rsplit('.', 1)[0]}_{ratio_decimal}.yaml"
    )

    # Insert config
    with open(
        f"{baseline_config_folder}/{baseline_insert_config_filename}",
        "r",
        encoding="utf-8",
    ) as f:
        insert_lines = f.readlines()

    insert_lines = update_config_lines(insert_lines, dataset_ratio)

    insert_cfg_path = f"{baseline_config_folder}/{insert_cfg_filename}"

    with open(insert_cfg_path, "w", encoding="utf-8") as f:
        f.writelines(insert_lines)

    print(f"Insert config saved in {insert_cfg_path}")

    # Query config
    with open(
        f"{baseline_config_folder}/{baseline_query_config_filename}",
        "r",
        encoding="utf-8",
    ) as f:
        query_lines = f.readlines()

    query_lines = update_config_lines(query_lines, dataset_ratio)

    query_cfg_path = f"{baseline_config_folder}/{query_cfg_filename}"

    with open(query_cfg_path, "w", encoding="utf-8") as f:
        f.writelines(query_lines)

    print(f"Query config saved in {query_cfg_path}")

    return {
        "insert": insert_cfg_path,
        "query": query_cfg_path,
    }


for ds_ratio in dataset_ratios:
    config_paths = gen_new_config_file(ds_ratio)

    insert_cmd = [
        "python3",
        f"{src_path}/run_new.py",
        "--config",
        config_paths["insert"],
        "--msys-config",
        baseline_insert_msys_filepath,
    ]


    query_cmd = [
        "python3",
        f"{src_path}/run_new.py",
        "--config",
        config_paths["query"],
        "--msys-config",
        baseline_query_msys_filepath,
    ]

    print(f"### Running insert benchmark for ratio = {ds_ratio} ###")
    subprocess.run(insert_cmd, check=True)
    print(f"### end of insertion for ratio = {ds_ratio}###")

    print(f"### Running query benchmark for ratio = {ds_ratio} ###")
    subprocess.run(query_cmd, check=True)
    print(f"### end of query for ratio = {ds_ratio}###")

    output_folder = "/local/amirk/RAGPerf/src/output"
    folders_all = os.listdir(output_folder)
    print("folders_all")
    print(folders_all)
    folders = [folder for folder in folders_all if folder.startswith("2026")]
    print("folders")
    print(folders)
    folders.sort()
    target_folder = folders[-1]
    print("target_folder")
    print(target_folder)
    os.rename(f"{output_folder}/{target_folder}", f"{output_folder}/lance_pdfimage_16_4_{str(ds_ratio).split('.')[-1]}")
    print(f"Logs generated in {output_folder}/lance_pdfimage_16_4_{str(ds_ratio).split('.')[-1]}")

