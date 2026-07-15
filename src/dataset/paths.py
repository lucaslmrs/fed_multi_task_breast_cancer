"""Filesystem layout of the datasets under ``data/``.

Every dataset owns one folder and follows the same internal convention, so the training code
never needs to know which dataset it is looking at -- switching datasets is a one-line change
to ``data.dataset`` in config.yaml:

    data/<dataset>/
        raw/          original download, extracted (only the preprocessing script reads it)
        archives/     original .zip archives
        <variant>/    preprocessed images/, masks/ and mapping.csv  <- what training reads
        federated/    federated_mapping.csv, the frozen master partition

``root``, ``dataset`` and ``variant`` all come from the ``data:`` section of config.yaml.
Resolving paths here rather than storing them in config keeps a single source of truth: every
setup (federated, standalone, centralized) derives the same partition file from the same key,
which is what makes the comparison experiment fair.
"""

from pathlib import Path

RAW_DIRNAME = "raw"
ARCHIVES_DIRNAME = "archives"
FEDERATED_DIRNAME = "federated"
MAPPING_FILENAME = "mapping.csv"
PARTITION_FILENAME = "federated_mapping.csv"


def dataset_dir(config_data: dict) -> Path:
    """Root folder of the dataset currently selected by ``data.dataset``."""
    return Path(config_data["root"]) / config_data["dataset"]


def raw_dir(config_data: dict) -> Path:
    """Untouched extracted download, consumed only by the preprocessing scripts."""
    return dataset_dir(config_data) / RAW_DIRNAME


def archives_dir(config_data: dict) -> Path:
    return dataset_dir(config_data) / ARCHIVES_DIRNAME


def processed_dir(config_data: dict) -> Path:
    """Preprocessed variant (``images/``, ``masks/``, ``mapping.csv``) that training reads."""
    return dataset_dir(config_data) / config_data["variant"]


def mapping_file(config_data: dict) -> Path:
    return processed_dir(config_data) / MAPPING_FILENAME


def partition_file(config_data: dict) -> Path:
    """Master federated partition CSV. Generated once by ``federated_partition`` and frozen."""
    return dataset_dir(config_data) / FEDERATED_DIRNAME / PARTITION_FILENAME


def require_mapping_file(config_data: dict) -> Path:
    path = mapping_file(config_data)
    if not path.exists():
        raise FileNotFoundError(
            f"Mapping file '{path}' not found. Preprocess '{config_data['dataset']}' first "
            f"(e.g. `python -m src.dataset.Curated_BUSI_preprocessing`).")
    return path


def require_partition_file(config_data: dict) -> Path:
    """Fail loudly instead of silently regenerating: every setup must share identical splits."""
    path = partition_file(config_data)
    if not path.exists():
        raise FileNotFoundError(
            f"Partition file '{path}' not found. Generate it once with "
            f"`python -m src.dataset.federated_partition` so all setups share identical splits.")
    return path
