import logging
import os
from typing import Dict, List

import yaml

from .models import TenantConfig

logger = logging.getLogger("registry")


class TenantRegistry:
    """
    Loads tenant configurations from a YAML file and provides
    thread-safe lookups by customer_id.

    Fail-fast: raises FileNotFoundError or ValueError at startup if
    the config is missing or a required field is absent.
    """

    def __init__(self, config_path: str):
        self._tenants: Dict[str, TenantConfig] = {}
        self._load_config(config_path)

    def _load_config(self, config_path: str) -> None:
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Tenant config not found: {config_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        for cid, raw in data.get("tenants", {}).items():
            # Sanitize optional fields that YAML may load as None
            raw.setdefault("jwks_url", None)
            raw["feature_flags"] = raw.get("feature_flags") or {}
            try:
                self._tenants[cid] = TenantConfig(**raw)
            except TypeError as exc:
                raise ValueError(f"Invalid config for tenant '{cid}': {exc}") from exc

        logger.info(f"[registry] loaded tenants={list(self._tenants.keys())}")

    def get_tenant(self, customer_id: str) -> TenantConfig:
        """Return TenantConfig or raise ValueError for unknown tenants."""
        if customer_id not in self._tenants:
            raise ValueError(f"Unknown tenant: '{customer_id}'")
        return self._tenants[customer_id]

    def list_tenants(self) -> List[str]:
        """Return sorted list of all registered customer IDs."""
        return sorted(self._tenants.keys())

    def list_all(self) -> List[TenantConfig]:
        """Return all TenantConfig objects (for health checks etc.)."""
        return list(self._tenants.values())
