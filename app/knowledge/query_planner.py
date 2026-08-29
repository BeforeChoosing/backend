"""Deterministic query planning for profile-grounded career retrieval.

The planner deliberately does not ask a language model to rewrite the query.
Confirmed ability cards are already structured data, so a small set of bounded
queries is easier to inspect, cache and reproduce than one long concatenated
prompt.  The resulting query list is sent to the embedding gateway in one
batch by :class:`HybridKnowledgeRetriever`.
"""

from __future__ import annotations

from collections.abc import Sequence

from app.schemas.profile import ProfileCard


MAX_CARD_QUERY_LENGTH = 360
QUERY_PLAN_VERSION = "career-rag-v2-multi-query"


def build_career_queries(
    cards: Sequence[ProfileCard],
    *,
    target_role: str,
) -> list[str]:
    """Build one role query plus one focused query per confirmed card.

    The first query anchors recall on the role.  Card queries preserve the
    user's confirmed wording and the next verification target, while omitting
    long evidence prose that tends to dilute semantic retrieval.  Duplicate
    queries are removed deterministically so repeated cards do not create
    extra embedding work.
    """

    role = " ".join(str(target_role).split()).strip()
    if not role:
        return []

    raw_queries = [f"{role} 岗位职责 能力要求 工作内容"]
    for card in cards:
        fields = [
            card.title,
            card.category,
            card.description,
            card.next_verification,
            card.workplace_application,
        ]
        compact = " ".join(" ".join(str(value).split()) for value in fields if value)
        compact = compact[:MAX_CARD_QUERY_LENGTH].strip()
        if compact:
            raw_queries.append(f"{role} {compact}")

    queries: list[str] = []
    seen: set[str] = set()
    for query in raw_queries:
        normalized = " ".join(query.split()).strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        queries.append(normalized)
    return queries
