from __future__ import annotations


class TenantIsolation:
    """Utility for enforcing same-tenant access at runtime."""

    def assert_same_tenant(self, context_tenant: str, requested_tenant: str) -> None:
        if context_tenant != requested_tenant:
            raise PermissionError('tenant isolation violation')

    def check_payload(self, payload: object, expected_tenant: str) -> object:
        """Best-effort check for dict-like payloads that carry tenant metadata."""
        if isinstance(payload, dict):
            payload_tenant = payload.get('tenant_id') or payload.get('tenant')
            if payload_tenant and payload_tenant != expected_tenant:
                raise PermissionError('tenant isolation violation in payload')
        return payload
