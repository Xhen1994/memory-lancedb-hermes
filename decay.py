"""Weibull lifecycle decay engine for lancedb-pro.

Implements the memory lifecycle model from memory-lancedb-pro:
- Core / Working / Peripheral tiers
- Weibull decay per tier
- Access reinforcement
"""

from __future__ import annotations

import logging
import math
import time
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Tier names
TIER_CORE = "core"
TIER_WORKING = "working"
TIER_PERIPHERAL = "peripheral"


class DecayEngine:
    """Weibull lifecycle decay engine.

    Computes a lifecycle composite score used for tier management and
    retrieval boosting. Higher = more important. Different Weibull beta
    per tier controls how aggressively importance decays.
    """

    def __init__(
        self,
        recency_half_life_days: float = 30.0,
        recency_weight: float = 0.4,
        frequency_weight: float = 0.3,
        intrinsic_weight: float = 0.3,
        importance_modulation: float = 1.5,
        beta_core: float = 0.8,
        beta_working: float = 1.0,
        beta_peripheral: float = 1.3,
        core_decay_floor: float = 0.9,
        working_decay_floor: float = 0.7,
        peripheral_decay_floor: float = 0.5,
    ):
        self.recency_half_life_days = recency_half_life_days
        self.recency_weight = recency_weight
        self.frequency_weight = frequency_weight
        self.intrinsic_weight = intrinsic_weight
        self.importance_modulation = importance_modulation
        self.beta_core = beta_core
        self.beta_working = beta_working
        self.beta_peripheral = beta_peripheral
        self.core_decay_floor = core_decay_floor
        self.working_decay_floor = working_decay_floor
        self.peripheral_decay_floor = peripheral_decay_floor

    def weibull_scale(self, half_life_days: float, beta: float) -> float:
        """Compute Weibull scale parameter from half-life and beta.

        We solve: 0.5 = exp(-(T_half/scale)^beta)
        => scale = T_half / (-ln(0.5))^(1/beta)
        """
        if half_life_days <= 0:
            return float("inf")
        return half_life_days / pow(-math.log(0.5), 1.0 / beta)

    def weibull_survival(
        self,
        age_days: float,
        half_life_days: float,
        beta: float,
    ) -> float:
        """Weibull survival probability: P(age) = exp(-(age/scale)^beta)."""
        if half_life_days <= 0 or age_days < 0:
            return 1.0
        scale = self.weibull_scale(half_life_days, beta)
        return math.exp(-pow(age_days / scale, beta))

    def compute_lifecycle(
        self,
        importance: float,
        confidence: float,
        access_count: int,
        age_days: float,
        tier: str = TIER_WORKING,
    ) -> float:
        """Compute lifecycle composite score (0-1).

        Formula (simplified from memory-lancedb-pro):
        score = recency_component + frequency_component + intrinsic_component

        Each weighted and normalized to [0, 1].
        """
        # Recency: Weibull survival
        beta = self._tier_beta(tier)
        half_life = self.recency_half_life_days
        recency_score = self.weibull_survival(age_days, half_life, beta)
        recency_norm = recency_score  # already 0-1

        # Frequency: log-scaled access count
        # 0 accesses = 0, 1 = ~0.5, 5+ = near 1.0
        freq_norm = min(1.0, math.log1p(access_count) / math.log(10))

        # Intrinsic: importance × confidence (modulated)
        intrinsic_raw = importance * confidence
        intrinsic_norm = min(1.0, intrinsic_raw * self.importance_modulation)

        # Weighted composite
        score = (
            self.recency_weight * recency_norm
            + self.frequency_weight * freq_norm
            + self.intrinsic_weight * intrinsic_norm
        ) / (
            self.recency_weight + self.frequency_weight + self.intrinsic_weight
        )

        return min(1.0, max(0.0, score))

    def apply_decay(
        self,
        importance: float,
        age_days: float,
        tier: str = TIER_WORKING,
    ) -> float:
        """Apply Weibull decay to importance, with tier-specific floors."""
        beta = self._tier_beta(tier)
        half_life = self.recency_half_life_days
        survival = self.weibull_survival(age_days, half_life, beta)
        floor = self._tier_decay_floor(tier)
        decayed = survival * importance
        return max(floor * importance, decayed)

    def _tier_beta(self, tier: str) -> float:
        beta_map = {
            TIER_CORE: self.beta_core,
            TIER_WORKING: self.beta_working,
            TIER_PERIPHERAL: self.beta_peripheral,
        }
        return beta_map.get(tier, self.beta_working)

    def _tier_decay_floor(self, tier: str) -> float:
        floor_map = {
            TIER_CORE: self.core_decay_floor,
            TIER_WORKING: self.working_decay_floor,
            TIER_PERIPHERAL: self.peripheral_decay_floor,
        }
        return floor_map.get(tier, self.working_decay_floor)

    def recency_boost(
        self,
        age_days: float,
        half_life_days: float = 14.0,
        max_boost: float = 0.1,
    ) -> float:
        """Additive recency bonus for retrieval scoring.

        Newer memories get an extra boost on top of their lifecycle score.
        At age=0: max_boost. At age=half_life_days: max_boost/2.
        At age=2*half_life: max_boost/4, etc.
        """
        if half_life_days <= 0 or age_days < 0:
            return 0.0
        return max_boost * pow(0.5, age_days / half_life_days)

    def time_decay_penalty(
        self,
        age_days: float,
        half_life_days: float = 60.0,
    ) -> float:
        """Multiplicative penalty for old memories in retrieval.

        Formula: 0.5 + 0.5 * exp(-age / half_life)
        At 0 days: ~1.0 (no penalty)
        At half_life: ~0.68
        At 2*half_life: ~0.59
        At 3*half_life: ~0.55
        """
        if half_life_days <= 0 or age_days <= 0:
            return 1.0
        return 0.5 + 0.5 * math.exp(-age_days / half_life_days)

    def access_reinforcement(
        self,
        current_half_life_days: float,
        access_count: int,
        reinforcement_factor: float = 0.5,
        max_multiplier: float = 3.0,
    ) -> float:
        """Extend half-life based on access frequency.

        Frequently accessed memories decay slower.
        Formula: half_life *= 1 + reinforcement_factor * log1p(access_count)
        Capped at max_multiplier × original half-life.
        """
        if reinforcement_factor <= 0 or access_count <= 0:
            return current_half_life_days
        multiplier = min(
            max_multiplier,
            1.0 + reinforcement_factor * math.log1p(access_count),
        )
        return current_half_life_days * multiplier

    def determine_tier(
        self,
        importance: float,
        confidence: float,
        access_count: int,
        age_days: float,
        current_tier: str = TIER_WORKING,
        core_access_threshold: int = 10,
        core_composite_threshold: float = 0.7,
        peripheral_composite_threshold: float = 0.15,
        peripheral_age_days: int = 60,
    ) -> str:
        """Determine which tier a memory belongs to based on lifecycle score."""
        composite = self.compute_lifecycle(
            importance=importance,
            confidence=confidence,
            access_count=access_count,
            age_days=age_days,
            tier=current_tier,
        )

        # Promote to core
        if composite >= core_composite_threshold and access_count >= core_access_threshold:
            return TIER_CORE

        # Demote to peripheral
        if (
            composite < peripheral_composite_threshold
            and age_days >= peripheral_age_days
        ):
            return TIER_PERIPHERAL

        # Otherwise keep working
        return TIER_WORKING
