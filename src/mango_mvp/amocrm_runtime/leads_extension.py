from __future__ import annotations

from mango_mvp.amocrm_runtime.amo_integration import fetch_lead_field_catalog, send_lead_custom_field_update
from mango_mvp.amocrm_runtime.deals import resolve_target_lead

__all__ = [
    "fetch_lead_field_catalog",
    "send_lead_custom_field_update",
    "resolve_target_lead",
]
