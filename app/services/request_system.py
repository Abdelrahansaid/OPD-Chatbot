"""Request capture for unanswered OPD chatbot questions."""
from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

from app.config import DATAVERSE_REQUESTS_TABLE, POWER_AUTOMATE_FLOW_URL

logger = logging.getLogger(__name__)

REQUESTS_FILE = Path("data/pending_requests.json")


class RequestSystem:
    def __init__(self, dataverse_client=None) -> None:
        self.dv = dataverse_client
        if self.dv is None:
            try:
                from app.services.dataverse_client import DataverseClient

                candidate = DataverseClient()
                self.dv = candidate if candidate.is_ready else None
            except Exception as exc:
                logger.debug("Dataverse client auto-init skipped: %s", exc)
        REQUESTS_FILE.parent.mkdir(exist_ok=True)
        if not REQUESTS_FILE.exists():
            REQUESTS_FILE.write_text("[]", encoding="utf-8")

    @property
    def use_dataverse(self) -> bool:
        return self.dv is not None and getattr(self.dv, "is_ready", False)

    def _load(self) -> list:
        try:
            return json.loads(REQUESTS_FILE.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return []

    def _save(self, data: list) -> None:
        REQUESTS_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def _replace_record(self, updated_record: Dict) -> None:
        records = self._load()
        for idx, record in enumerate(records):
            if record.get("request_id") == updated_record.get("request_id"):
                records[idx] = updated_record
                self._save(records)
                return
        records.append(updated_record)
        self._save(records)

    @staticmethod
    def _repair_mojibake(text: Optional[str]) -> Optional[str]:
        if not text or not any(marker in text for marker in ("Ø", "Ù", "â")):
            return text
        try:
            return text.encode("latin1").decode("utf-8")
        except UnicodeError:
            return text

    def check_similar(self, question: str) -> Optional[Dict]:
        q_words = set(re.findall(r"\w+", question.lower()))
        if not q_words:
            return None

        for rec in self._load():
            existing_words = set(re.findall(r"\w+", rec.get("question", "").lower()))
            overlap = len(q_words & existing_words) / max(len(q_words), 1)
            if overlap > 0.6:
                return rec
        return None

    def submit(self, question: str, user_role: str = "General User") -> Dict:
        existing = self.check_similar(question)
        if existing:
            return {
                "status": "duplicate",
                "request_id": existing["request_id"],
                "message": (
                    f"A similar request already exists (ID: {existing['request_id']}, "
                    f"Status: {existing['status']}). The team is already working on it."
                ),
            }

        records = self._load()
        req_id = f"REQ-{len(records) + 1:04d}"
        created_at = datetime.now().isoformat(timespec="seconds")
        record = {
            "request_id": req_id,
            "question": question,
            "user_role": user_role,
            "status": "Pending",
            "created_at": created_at,
            "category": self._guess_category(question),
            "remote_channel": "power_automate_flow" if POWER_AUTOMATE_FLOW_URL else "local_only",
            "remote_sync_status": "not_configured" if not POWER_AUTOMATE_FLOW_URL else "pending",
            "remote_error": None,
        }

        # Safety layer: persist locally before any network call.
        records.append(record)
        self._save(records)

        flow_result = self._send_to_power_automate(record)
        if flow_result["ok"]:
            flow_status = flow_result.get("flow_status") or "synced"
            record["remote_sync_status"] = "duplicate_in_dataverse" if flow_status == "exists" else "synced"
            record["remote_response_status"] = flow_status
            flow_message = self._repair_mojibake(flow_result.get("message"))
            record["remote_response_message"] = flow_message
            record["remote_error"] = None
            self._replace_record(record)

            if flow_status == "exists":
                return {
                    "status": "duplicate",
                    "request_id": req_id,
                    "message": flow_message or (
                        f"A similar request already exists in Dataverse. (ID: **{req_id}**)"
                    ),
                }

            return {
                "status": "submitted",
                "request_id": req_id,
                "message": flow_message or (
                    f"Request submitted successfully! (ID: **{req_id}**)\n"
                    "Saved locally and sent to Dataverse via Power Automate."
                ),
            }

        record["remote_sync_status"] = flow_result["status"]
        record["remote_error"] = flow_result.get("error")
        self._replace_record(record)

        # Legacy direct Dataverse path stays optional for older deployments.
        if not POWER_AUTOMATE_FLOW_URL and self.use_dataverse:
            dv_record = {
                "opd_request_id": req_id,
                "opd_question": question,
                "opd_user_role": user_role,
                "opd_status": "Pending",
                "opd_category": record["category"],
                "opd_created_at": created_at,
            }
            try:
                self.dv.create_record(DATAVERSE_REQUESTS_TABLE, dv_record)
            except Exception as exc:
                logger.warning("Dataverse request sync failed; local copy retained: %s", exc)

        return {
            "status": "submitted_locally",
            "request_id": req_id,
            "message": (
                f"Request saved locally! (ID: **{req_id}**)\n"
                "Power Automate sync is pending or not configured, so the request was kept safely in JSON."
            ),
        }

    def _send_to_power_automate(self, record: Dict) -> Dict:
        if not POWER_AUTOMATE_FLOW_URL:
            return {"ok": False, "status": "not_configured", "error": "POWER_AUTOMATE_FLOW_URL is not set."}

        try:
            import requests
        except ImportError as exc:
            return {"ok": False, "status": "pending_flow", "error": str(exc)}

        payload = {
            "request_id": record["request_id"],
            "question": record["question"],
            "user_role": record["user_role"],
            "status": record["status"],
            "category": record["category"],
            "created_at": record["created_at"],
        }

        try:
            response = requests.post(
                POWER_AUTOMATE_FLOW_URL,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=15,
            )
            response.raise_for_status()
            try:
                flow_response = response.json()
            except ValueError:
                flow_response = {"status": "synced", "message": response.text.strip()}

            flow_status = flow_response.get("status") or "synced"
            logger.info(
                "Request %s sent to Power Automate. Flow status: %s",
                record["request_id"],
                flow_status,
            )
            return {
                "ok": True,
                "status": "synced",
                "flow_status": flow_status,
                "message": flow_response.get("message"),
                "error": None,
            }
        except requests.RequestException as exc:
            logger.warning("Power Automate sync failed for %s: %s", record["request_id"], exc)
            return {"ok": False, "status": "pending_flow", "error": str(exc)}

    def _guess_category(self, question: str) -> str:
        q = question.lower()
        if any(w in q for w in ["formula", "معادلة", "definition", "تعريف"]):
            return "KPI Definition"
        if any(w in q for w in ["specialty", "تخصص"]):
            return "New Dimension"
        if any(w in q for w in ["forecast", "predict", "تنبؤ"]):
            return "Forecasting"
        if any(w in q for w in ["insurance", "rejection", "تأمين", "رفض"]):
            return "Billing System"
        return "Analytics Feature"


def submit_request(question: str, role: str = "General User", dataverse_client=None) -> Dict:
    return RequestSystem(dataverse_client=dataverse_client).submit(question, user_role=role)
