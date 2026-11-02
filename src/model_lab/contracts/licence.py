"""Licence status vocabulary shared by the source registry and every
dataset/option snapshot.

Per the operator's ruling and MODEL1_foundation_contracts.md §9 rule 4: a
snapshot with `licence_status in {UNRESOLVED, RESTRICTED}` is stored
`quarantined` and can feed only experiments tagged `research_only_unlicensed`,
which can never produce promotion evidence. Authority is never upgraded
silently — `derive_research_state` is the single mechanical function that
performs this mapping; nothing else in the codebase should hand-assign a
research state.
"""

from __future__ import annotations

from enum import StrEnum


class LicenceStatus(StrEnum):
    """Assigned to a *source* (a provider/series), per the R1b terms audit."""

    CLEAR = "CLEAR"
    UNRESOLVED = "UNRESOLVED"
    RESTRICTED = "RESTRICTED"


class ResearchState(StrEnum):
    """Derived, mechanically, from a source's LicenceStatus. Never
    hand-assigned."""

    PROMOTION_ELIGIBLE = "promotion_eligible"
    RESEARCH_ONLY_UNLICENSED = "research_only_unlicensed"
    QUARANTINED = "quarantined"


def derive_research_state(licence_status: LicenceStatus) -> ResearchState:
    """The one mechanical licence -> research-state mapping.

    CLEAR -> promotion_eligible (still subject to every other data-quality
             gate; this function only clears the licence gate).
    UNRESOLVED -> research_only_unlicensed (usable for method development,
             never for promotion evidence).
    RESTRICTED -> quarantined (a RESTRICTED source's data is stored, honestly
             labelled, but treated as unusable even for research analyses
             that might be shown externally; still permitted for internal
             method-development plumbing tests only, never for any claim).
    """
    if licence_status == LicenceStatus.CLEAR:
        return ResearchState.PROMOTION_ELIGIBLE
    if licence_status == LicenceStatus.UNRESOLVED:
        return ResearchState.RESEARCH_ONLY_UNLICENSED
    if licence_status == LicenceStatus.RESTRICTED:
        return ResearchState.QUARANTINED
    raise ValueError(f"Unhandled LicenceStatus: {licence_status!r}")


def can_produce_promotion_evidence(research_state: ResearchState) -> bool:
    return research_state == ResearchState.PROMOTION_ELIGIBLE
