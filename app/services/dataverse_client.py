"""Small Dataverse Web API client used by the request system.

The client is intentionally optional: if credentials are missing or required
packages are not installed, `is_ready` stays False and callers can fall back to
local storage.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from app.config import (
    DATAVERSE_CLIENT_ID,
    DATAVERSE_CLIENT_SECRET,
    DATAVERSE_TENANT_ID,
    DATAVERSE_URL,
)

logger = logging.getLogger(__name__)


class DataverseClient:
    def __init__(
        self,
        url: Optional[str] = None,
        tenant_id: Optional[str] = None,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
    ) -> None:
        self.url = (url or DATAVERSE_URL or "").rstrip("/")
        self.tenant_id = tenant_id or DATAVERSE_TENANT_ID
        self.client_id = client_id or DATAVERSE_CLIENT_ID
        self.client_secret = client_secret or DATAVERSE_CLIENT_SECRET

    @property
    def is_ready(self) -> bool:
        return all([self.url, self.tenant_id, self.client_id, self.client_secret])

    def _get_token(self) -> str:
        if not self.is_ready:
            raise RuntimeError("Dataverse settings are incomplete.")

        try:
            import msal
        except ImportError as exc:
            raise RuntimeError("Package 'msal' is required for Dataverse integration.") from exc

        authority = f"https://login.microsoftonline.com/{self.tenant_id}"
        app = msal.ConfidentialClientApplication(
            self.client_id,
            authority=authority,
            client_credential=self.client_secret,
        )
        token = app.acquire_token_for_client(scopes=[f"{self.url}/.default"])
        if "access_token" not in token:
            raise RuntimeError(f"Dataverse token request failed: {token.get('error_description') or token}")
        return token["access_token"]

    def create_record(self, table_name: str, record: Dict[str, Any]) -> Dict[str, Any]:
        try:
            import requests
        except ImportError as exc:
            raise RuntimeError("Package 'requests' is required for Dataverse integration.") from exc

        endpoint = f"{self.url}/api/data/v9.2/{table_name}"
        headers = {
            "Authorization": f"Bearer {self._get_token()}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "OData-MaxVersion": "4.0",
            "OData-Version": "4.0",
        }
        response = requests.post(endpoint, json=record, headers=headers, timeout=30)
        response.raise_for_status()
        return {"status": "created", "location": response.headers.get("OData-EntityId", "")}
