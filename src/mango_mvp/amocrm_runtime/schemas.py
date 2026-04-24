from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel


class AmoIntegrationSetupRead(BaseModel):
    integration_mode: str
    redirect_uri: Optional[str]
    secrets_uri: Optional[str]
    scopes: list[str]
    integration_name: str
    integration_description: str
    logo_url: Optional[str]
    account_base_url_hint: Optional[str]
    button_snippet: Optional[str]


class AmoIntegrationStatusRead(AmoIntegrationSetupRead):
    connected: bool
    status: str
    account_base_url: Optional[str]
    account_subdomain: Optional[str]
    client_id_present: bool
    client_secret_present: bool
    access_token_present: bool
    refresh_token_present: bool
    authorized_at: Optional[datetime]
    expires_at: Optional[datetime]
    last_error: Optional[str]
    contact_field_catalog_synced_at: Optional[datetime]
    contact_field_count: int
    required_contact_fields_present: list[str]
    required_contact_fields_missing: list[str]
    lead_field_count: int
    required_lead_fields_present: list[str]
    required_lead_fields_missing: list[str]
    lead_field_sync_error: Optional[str]
    token_source: Optional[str]


class AmoIntegrationSecretsWebhookResponse(BaseModel):
    status: str
    summary: str


class AmoIntegrationRefreshResponse(BaseModel):
    status: str
    summary: str
    expires_at: Optional[datetime]


class AmoContactFieldSyncResponse(BaseModel):
    status: str
    summary: str
    field_count: int
    synced_at: Optional[datetime]
