"""ADIP configuration loader.

Loads and provides access to store metadata and model hyperparameters
from YAML configuration files. Implements singleton pattern for
efficient reuse across the application.
"""

from pathlib import Path
from typing import Optional

import yaml

# Project root is three levels up from this file:
# src/utils/config.py -> src/utils -> src -> project_root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

_DEFAULT_STORES_PATH = _PROJECT_ROOT / "configs" / "stores.yaml"
_DEFAULT_MODELS_PATH = _PROJECT_ROOT / "configs" / "models.yaml"


class ADIPConfig:
    """Central configuration object for the ADIP platform.

    Loads store metadata from ``configs/stores.yaml`` and model
    hyperparameters from ``configs/models.yaml``.  A singleton
    instance is available via :meth:`get_instance`.

    Attributes:
        stores_config: Parsed contents of stores.yaml.
        models_config: Parsed contents of models.yaml.
    """

    _instance: Optional["ADIPConfig"] = None

    def __init__(
        self,
        stores_path: Optional[str] = None,
        models_path: Optional[str] = None,
    ) -> None:
        """Initialise the configuration by loading YAML files.

        Args:
            stores_path: Override path for stores.yaml.  Defaults to
                ``<project_root>/configs/stores.yaml``.
            models_path: Override path for models.yaml.  Defaults to
                ``<project_root>/configs/models.yaml``.

        Raises:
            FileNotFoundError: If either configuration file is missing.
            yaml.YAMLError: If either file contains invalid YAML.
        """
        stores_file = Path(stores_path) if stores_path else _DEFAULT_STORES_PATH
        models_file = Path(models_path) if models_path else _DEFAULT_MODELS_PATH

        if not stores_file.exists():
            raise FileNotFoundError(f"Stores config not found: {stores_file}")
        if not models_file.exists():
            raise FileNotFoundError(f"Models config not found: {models_file}")

        with open(stores_file, "r", encoding="utf-8") as fh:
            self.stores_config: dict = yaml.safe_load(fh)

        with open(models_file, "r", encoding="utf-8") as fh:
            self.models_config: dict = yaml.safe_load(fh)

    # ------------------------------------------------------------------
    # Singleton accessor
    # ------------------------------------------------------------------

    @classmethod
    def get_instance(
        cls,
        stores_path: Optional[str] = None,
        models_path: Optional[str] = None,
    ) -> "ADIPConfig":
        """Return the singleton ADIPConfig instance, creating it if needed.

        Args:
            stores_path: Override path for stores.yaml (only used on first call).
            models_path: Override path for models.yaml (only used on first call).

        Returns:
            The singleton ADIPConfig instance.
        """
        if cls._instance is None:
            cls._instance = cls(stores_path=stores_path, models_path=models_path)
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton — primarily for testing."""
        cls._instance = None

    # ------------------------------------------------------------------
    # Store helpers
    # ------------------------------------------------------------------

    @property
    def stores(self) -> dict:
        """Return the full store metadata dictionary keyed by store ID.

        Returns:
            Dict mapping integer store IDs to their metadata dicts.
        """
        return self.stores_config.get("stores", {})

    @property
    def store_id_map(self) -> dict:
        """Return the store-name-to-ID mapping.

        Returns:
            Dict mapping store name strings to integer IDs.
        """
        return self.stores_config.get("store_id_map", {})

    def get_store_name(self, store_id: int) -> str:
        """Look up the human-readable name for a store.

        Args:
            store_id: Numeric store identifier (e.g. 3, 5, 8).

        Returns:
            Store name string.

        Raises:
            KeyError: If store_id is not found in the configuration.
        """
        store = self.stores.get(store_id)
        if store is None:
            raise KeyError(f"Unknown store_id: {store_id}")
        return store["name"]

    def get_store_ids(self) -> list[int]:
        """Return a sorted list of all configured store IDs.

        Returns:
            Sorted list of integer store IDs.
        """
        return sorted(int(k) for k in self.stores.keys())

    # ------------------------------------------------------------------
    # Model helpers
    # ------------------------------------------------------------------

    @property
    def models(self) -> dict:
        """Return the full model hyperparameter dictionary.

        Returns:
            Dict mapping model name strings to their config dicts.
        """
        return self.models_config.get("models", {})

    @property
    def evaluation_thresholds(self) -> dict:
        """Return evaluation threshold settings.

        Returns:
            Dict with keys like ``mape_pass``, ``mape_review``.
        """
        eval_cfg = self.models_config.get("evaluation", {})
        return eval_cfg.get("thresholds", {})

    @property
    def walk_forward_config(self) -> dict:
        """Return walk-forward cross-validation settings.

        Returns:
            Dict with keys ``n_splits`` and ``test_days``.
        """
        eval_cfg = self.models_config.get("evaluation", {})
        return eval_cfg.get("walk_forward", {})

    @property
    def holdout_days(self) -> int:
        """Return the number of holdout days for final evaluation.

        Returns:
            Integer number of holdout days.
        """
        eval_cfg = self.models_config.get("evaluation", {})
        return int(eval_cfg.get("holdout_days", 30))

    def get_model_config(self, model_name: str) -> dict:
        """Retrieve configuration for a specific model.

        Args:
            model_name: One of ``sarima``, ``prophet``, ``xgboost``,
                ``ensemble``.

        Returns:
            Dict of hyperparameters for the requested model.

        Raises:
            KeyError: If the model_name is not found.
        """
        cfg = self.models.get(model_name)
        if cfg is None:
            raise KeyError(f"Unknown model '{model_name}'. " f"Available: {list(self.models.keys())}")
        return cfg

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        store_ids = self.get_store_ids()
        model_names = list(self.models.keys())
        return f"ADIPConfig(stores={store_ids}, models={model_names})"
