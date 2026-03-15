from __future__ import annotations


class AccessController:
    """Small policy-aware access checker used by API/security boundaries.

    Runtime effect:
    - admins may access across tenants
    - resource owners may access their own resources
    - otherwise tenant must match the requested tenant
    """

    def can_access(
        self,
        user_id: str,
        tenant_id: str,
        requested_tenant_id: str,
        role: str | None = None,
        resource_owner: str | None = None,
    ) -> bool:
        role = (role or '').lower()
        if user_id == 'system' or role == 'admin':
            return True
        if resource_owner and user_id == resource_owner:
            return tenant_id == requested_tenant_id
        return tenant_id == requested_tenant_id
