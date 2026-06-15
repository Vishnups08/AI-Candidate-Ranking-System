"""
Stage 4: Behavioral Signal Multiplier.
Applied on top of feature scores. Range: 0.6-1.3.
A perfect-on-paper candidate who's inactive gets ~0.65x; an engaged one gets ~1.25x.
"""

import math
from datetime import datetime, date

import config


def compute_behavioral_multiplier(candidate: dict) -> float:
    """
    Compute behavioral multiplier for a candidate.
    Returns a value in [BEHAVIORAL_MULTIPLIER_MIN, BEHAVIORAL_MULTIPLIER_MAX].
    """
    signals = candidate.get("redrob_signals", {})

    # Component 1: Availability (25%)
    avail = _availability_score(signals)

    # Component 2: Engagement quality (25%)
    engage = _engagement_score(signals)

    # Component 3: Platform credibility (20%)
    cred = _credibility_score(signals)

    # Component 4: Market signal (15%)
    market = _market_score(signals)

    # Component 5: Technical engagement (15%)
    tech = _tech_score(signals)

    # Weighted raw score (0-1)
    raw = (
        avail * 0.25
        + engage * 0.25
        + cred * 0.20
        + market * 0.15
        + tech * 0.15
    )

    # Map to multiplier range
    multiplier = (
        config.BEHAVIORAL_MULTIPLIER_MIN
        + raw * (config.BEHAVIORAL_MULTIPLIER_MAX - config.BEHAVIORAL_MULTIPLIER_MIN)
    )

    return multiplier


def compute_behavioral_additive(candidate: dict) -> float:
    """
    Compute small additive bonus for very strong behavioral signals.
    Returns 0.0-0.05.
    """
    signals = candidate.get("redrob_signals", {})

    # Only give bonus for very strong signals
    bonus = 0.0

    # Open to work + high response rate + recent activity
    if signals.get("open_to_work_flag", False):
        if signals.get("recruiter_response_rate", 0) > 0.7:
            bonus += 0.02

    # High platform engagement
    if signals.get("saved_by_recruiters_30d", 0) > 10:
        bonus += 0.01

    # Strong github activity
    github = signals.get("github_activity_score", -1)
    if github > 70:
        bonus += 0.01

    # All verified
    if (signals.get("verified_email", False)
            and signals.get("verified_phone", False)
            and signals.get("linkedin_connected", False)):
        bonus += 0.01

    return min(bonus, config.BEHAVIORAL_ADDITIVE_WEIGHT)


def _availability_score(signals: dict) -> float:
    """Score based on whether candidate is actually available."""
    components = []

    # Open to work flag
    open_to_work = signals.get("open_to_work_flag", False)
    components.append(1.0 if open_to_work else 0.3)

    # Last active date (recency)
    last_active = signals.get("last_active_date", "")
    if last_active:
        try:
            last_active_date = datetime.strptime(last_active, "%Y-%m-%d").date()
            days_since = (date.today() - last_active_date).days
            if days_since <= 7:
                recency = 1.0
            elif days_since <= 30:
                recency = 0.9
            elif days_since <= 90:
                recency = 0.7
            elif days_since <= 180:
                recency = 0.4
            else:
                recency = 0.15  # JD: "hasn't logged in for 6 months"
        except ValueError:
            recency = 0.5
    else:
        recency = 0.5

    components.append(recency)

    # Recruiter response rate
    response_rate = signals.get("recruiter_response_rate", 0.5)
    components.append(response_rate)

    return sum(components) / len(components)


def _engagement_score(signals: dict) -> float:
    """Score based on engagement quality."""
    components = []

    # Interview completion rate
    interview_rate = signals.get("interview_completion_rate", 0.5)
    components.append(interview_rate)

    # Profile completeness
    completeness = signals.get("profile_completeness_score", 50) / 100.0
    components.append(completeness)

    # Response time (lower is better)
    avg_response_hours = signals.get("avg_response_time_hours", 72)
    if avg_response_hours <= 12:
        response_time_score = 1.0
    elif avg_response_hours <= 24:
        response_time_score = 0.9
    elif avg_response_hours <= 48:
        response_time_score = 0.7
    elif avg_response_hours <= 96:
        response_time_score = 0.5
    elif avg_response_hours <= 168:
        response_time_score = 0.3
    else:
        response_time_score = 0.15
    components.append(response_time_score)

    # Applications submitted (shows active searching)
    apps = signals.get("applications_submitted_30d", 0)
    if apps >= 5:
        components.append(0.9)
    elif apps >= 2:
        components.append(0.7)
    elif apps >= 1:
        components.append(0.5)
    else:
        components.append(0.3)

    return sum(components) / len(components)


def _credibility_score(signals: dict) -> float:
    """Score based on verification and trust signals."""
    score = 0.0

    if signals.get("verified_email", False):
        score += 0.3
    if signals.get("verified_phone", False):
        score += 0.3
    if signals.get("linkedin_connected", False):
        score += 0.4

    return score


def _market_score(signals: dict) -> float:
    """Score based on external market validation."""
    # Saved by recruiters (log-scaled)
    saved = signals.get("saved_by_recruiters_30d", 0)
    saved_score = min(math.log1p(saved) / math.log1p(20), 1.0)

    # Search appearances (log-scaled)
    appearances = signals.get("search_appearance_30d", 0)
    appearance_score = min(math.log1p(appearances) / math.log1p(200), 1.0)

    # Profile views (log-scaled)
    views = signals.get("profile_views_received_30d", 0)
    view_score = min(math.log1p(views) / math.log1p(30), 1.0)

    return (saved_score * 0.4 + appearance_score * 0.3 + view_score * 0.3)


def _tech_score(signals: dict) -> float:
    """Score based on technical engagement signals."""
    # GitHub activity
    github = signals.get("github_activity_score", -1)
    if github < 0:
        github_score = 0.4  # No GitHub linked — neutral, not penalized heavily
    else:
        github_score = min(github / 80.0, 1.0)

    # Offer acceptance rate (shows reliability)
    offer_rate = signals.get("offer_acceptance_rate", -1)
    if offer_rate < 0:
        offer_score = 0.5  # No history — neutral
    else:
        offer_score = offer_rate

    # Connection count (network strength)
    connections = signals.get("connection_count", 0)
    connection_score = min(math.log1p(connections) / math.log1p(500), 1.0)

    return github_score * 0.4 + offer_score * 0.3 + connection_score * 0.3
