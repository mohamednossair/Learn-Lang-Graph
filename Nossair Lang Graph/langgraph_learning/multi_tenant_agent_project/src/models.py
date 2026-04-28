from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass(frozen=True)
class TenantConfig:
    customer_id: str
    display_name: str
    solr_core: str
    s3_prefix: str
    db_pool_name: str
    allowed_topics: List[str]
    blocked_topics: List[str]
    out_of_scope_msg: str
    custom_instructions: str
    jwks_url: Optional[str] = None
    feature_flags: Dict[str, bool] = field(default_factory=dict)

    def has_feature(self, flag: str) -> bool:
        """Check if a feature flag is enabled for this tenant."""
        return self.feature_flags.get(flag, False)
