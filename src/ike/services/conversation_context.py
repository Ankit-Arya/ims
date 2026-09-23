from __future__ import annotations

from uuid import UUID

from ike.retrieval.query_plan import QueryPlan


class ConversationContextStore:
    """Deprecated compatibility shim. IMS Q&A is stateless across requests.

    Cross-query context was removed in IMS 0.10.1. These no-op methods remain only so
    an out-of-tree integration importing the old symbol cannot accidentally persist or
    re-inject prior query scope. New application code must not depend on this class.
    """

    def get(self, user_id: UUID) -> dict[str, str]:
        del user_id
        return {}

    def update_from_plan(self, user_id: UUID, plan: QueryPlan) -> dict[str, str]:
        del user_id, plan
        return {}

    def clear(self, user_id: UUID) -> None:
        del user_id


conversation_context = ConversationContextStore()
