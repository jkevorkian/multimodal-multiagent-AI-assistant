from __future__ import annotations

from dataclasses import dataclass

from app.contracts.steering import SteeringRequest

_ALLOWED_PROFILES = {"balanced", "concise", "strict-grounded", "creative"}


@dataclass(frozen=True)
class SteeringResolution:
    profile: str
    notes: list[str]


def resolve_profile(steering: SteeringRequest | None) -> SteeringResolution:
    requested = (steering.profile if steering else "balanced").strip().lower()
    if requested not in _ALLOWED_PROFILES:
        return SteeringResolution(profile="balanced", notes=[f"invalid_profile:{requested}:fallback=balanced"])
    return SteeringResolution(profile=requested, notes=[])


def apply_tool_policy(
    *,
    available_tools: list[str],
    requested_tools: list[str] | None,
    steering: SteeringRequest | None,
) -> tuple[list[str], list[str]]:
    notes: list[str] = []
    selected = requested_tools or available_tools
    selected = [tool for tool in selected if tool in available_tools]

    if not steering or steering.tools is None:
        return selected, notes

    policy = steering.tools
    if policy.allow:
        allowed = set(policy.allow)
        selected = [tool for tool in selected if tool in allowed]
        notes.append(f"tool_allow_applied:{sorted(allowed)}")
    if policy.deny:
        denied = set(policy.deny)
        selected = [tool for tool in selected if tool not in denied]
        notes.append(f"tool_deny_applied:{sorted(denied)}")
    if policy.require:
        required = [tool for tool in policy.require if tool in available_tools]
        missing = [tool for tool in required if tool not in selected]
        selected.extend(missing)
        if required:
            notes.append(f"tool_require_applied:{required}")
    return sorted(set(selected)), notes


def apply_answer_style(answer: str, profile: str) -> str:
    normalized = profile.strip().lower()
    if normalized == "concise":
        compact = " ".join(answer.split())
        return compact[:280].rstrip() + ("..." if len(compact) > 280 else "")
    if normalized == "creative":
        return f"Creative framing: {answer}"
    return answer


def enforce_grounding_policy(
    *,
    answer: str,
    citations: list[str],
    profile: str,
    steering: SteeringRequest | None,
) -> tuple[str, list[str]]:
    notes: list[str] = []
    min_citations = 0
    abstain = False

    if profile == "strict-grounded":
        min_citations = max(min_citations, 1)
        abstain = True
    if steering and steering.grounding is not None:
        min_citations = max(min_citations, int(steering.grounding.min_citations))
        abstain = bool(steering.grounding.abstain_on_insufficient_citations) or abstain

    if len(citations) >= min_citations:
        return answer, notes
    notes.append(f"grounding_shortfall:{len(citations)}<{min_citations}")
    if abstain:
        return (
            "I do not have enough grounded citations to answer under the current steering policy.",
            notes,
        )
    return answer, notes
