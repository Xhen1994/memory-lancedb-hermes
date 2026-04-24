"""Config loader for lancedb-pro memory provider.

Loads config from $HERMES_HOME/lancedb_pro.json with environment variable overrides.
Mirrors the SiliconFlow/SiliconFlow config structure from memory-lancedb-pro.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from hermes_constants import get_hermes_home


def _load_json_config() -> Dict[str, Any]:
    """Load base config from lancedb_pro.json."""
    cfg_path = get_hermes_home() / "lancedb_pro.json"
    if not cfg_path.exists():
        return {}
    try:
        return json.loads(cfg_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _env_override(base: Dict[str, Any], key: str, env_var: str,
                  default: Any = None) -> Any:
    """Return env var if set, else base.get(key) or default."""
    if env_var in os.environ and os.environ[env_var]:
        return os.environ[env_var]
    return base.get(key, default)


class LanceDBProConfig:
    """Unified config object — all settings with smart defaults from memory-lancedb-pro."""

    def __init__(self, cfg: Optional[Dict[str, Any]] = None):
        cfg = cfg or _load_json_config()

        # ── Embedding ────────────────────────────────────────────────
        emb = cfg.get("embedding", {})
        self.emb_provider = emb.get("provider", "openai-compatible")
        self.emb_api_key = _env_override(emb, "apiKey", "EMBEDDING_API_KEY", "")
        self.emb_base_url = emb.get("baseURL", "https://api.siliconflow.cn/v1")
        self.emb_model = emb.get("model", "BAAI/bge-m3")
        self.emb_dimensions = int(emb.get("dimensions", 1024))
        self.emb_normalized = bool(emb.get("normalized", False))
        self.emb_task_query = emb.get("taskQuery", "")
        self.emb_task_passage = emb.get("taskPassage", "")
        self.emb_chunking = bool(emb.get("chunking", True))
        # Extended fields from memory-lancedb-pro
        self.emb_request_dimensions = int(emb["requestDimensions"]) if emb.get("requestDimensions") else None
        self.emb_omit_dimensions = bool(emb.get("omitDimensions", False))
        self.emb_api_version = emb.get("apiVersion", "")  # Azure only

        # ── Rerank ───────────────────────────────────────────────────
        rer = cfg.get("rerank", {})
        self.rerank_enabled = bool(rer.get("enabled", True))
        self.rerank_provider = rer.get("provider", "siliconflow")
        self.rerank_api_key = _env_override(rer, "apiKey", "RERANK_API_KEY",
                                             self.emb_api_key)
        self.rerank_endpoint = rer.get("endpoint",
                                        "https://api.siliconflow.cn/v1/rerank")
        self.rerank_model = rer.get("model", "BAAI/bge-reranker-v2-m3")
        self.rerank_timeout_ms = int(rer.get("timeoutMs", 5000))

        # ── LLM (for smart extraction) ───────────────────────────────
        llm = cfg.get("llm", {})
        self.llm_api_key = _env_override(llm, "apiKey", "LLM_API_KEY",
                                          self.emb_api_key)
        self.llm_base_url = llm.get("baseURL",
                                     "https://api.minimaxi.com/anthropic/v1")
        self.llm_model = llm.get("model", "MiniMax-M2.7")
        self.llm_timeout_ms = int(llm.get("timeoutMs", 60000))

        # ── Retrieval ───────────────────────────────────────────────
        ret = cfg.get("retrieval", {})
        self.ret_mode = ret.get("mode", "hybrid")
        self.ret_vector_weight = float(ret.get("vectorWeight", 0.7))
        self.ret_bm25_weight = float(ret.get("bm25Weight", 0.3))
        self.ret_min_score = float(ret.get("minScore", 0.3))
        self.ret_hard_min_score = float(ret.get("hardMinScore", 0.35))
        self.ret_candidate_pool_size = int(ret.get("candidatePoolSize", 20))
        self.ret_recency_half_life_days = float(
            ret.get("recencyHalfLifeDays", 14))
        self.ret_recency_weight = float(ret.get("recencyWeight", 0.1))
        self.ret_filter_noise = bool(ret.get("filterNoise", True))
        self.ret_length_norm_anchor = int(ret.get("lengthNormAnchor", 500))
        self.ret_time_decay_half_life_days = float(
            ret.get("timeDecayHalfLifeDays", 60))
        self.ret_reinforcement_factor = float(
            ret.get("reinforcementFactor", 0.5))
        self.ret_max_half_life_multiplier = float(
            ret.get("maxHalfLifeMultiplier", 3))

        # ── Auto Capture / Recall ────────────────────────────────────
        self.auto_capture = bool(cfg.get("autoCapture", True))
        self.auto_recall = bool(cfg.get("autoRecall", True))
        self.capture_assistant = bool(cfg.get("captureAssistant", False))
        self.smart_extraction = bool(cfg.get("smartExtraction", True))
        self.extract_min_messages = int(cfg.get("extractMinMessages", 2))
        self.extract_max_chars = int(cfg.get("extractMaxChars", 8000))
        self.recall_max_chars = int(cfg.get("recallMaxChars", 600))
        self.recall_max_items = int(cfg.get("recallMaxItems", 3))
        self.recall_mode = cfg.get("recallMode", "full")
        self.max_recall_per_turn = int(cfg.get("maxRecallPerTurn", 10))
        # Fields from memory-lancedb-pro
        self.auto_recall_min_length = int(cfg.get("autoRecallMinLength", 15))
        self.auto_recall_min_repeated = int(cfg.get("autoRecallMinRepeated", 0))
        self.auto_recall_max_query_length = int(cfg.get("autoRecallMaxQueryLength", 2000))
        self.auto_recall_timeout_ms = int(cfg.get("autoRecallTimeoutMs", 5000))
        self.auto_recall_per_item_max_chars = int(cfg.get("autoRecallPerItemMaxChars", 180))
        self.auto_recall_max_chars_total = int(cfg.get("autoRecallMaxChars", 600))
        self.recall_exclude_agents: List[str] = list(cfg.get("autoRecallExcludeAgents", []))
        self.recall_include_agents: List[str] = list(cfg.get("autoRecallIncludeAgents", []))

        # ── Decay ───────────────────────────────────────────────────
        dec = cfg.get("decay", {})
        self.decay_recency_half_life_days = float(
            dec.get("recencyHalfLifeDays", 30))
        self.decay_frequency_weight = float(dec.get("frequencyWeight", 0.3))
        self.decay_intrinsic_weight = float(dec.get("intrinsicWeight", 0.3))
        self.decay_recency_weight = float(dec.get("recencyWeight", 0.4))
        self.decay_beta_core = float(dec.get("betaCore", 0.8))
        self.decay_beta_working = float(dec.get("betaWorking", 1.0))
        self.decay_beta_peripheral = float(dec.get("betaPeripheral", 1.3))
        self.decay_importance_modulation = float(
            dec.get("importanceModulation", 1.5))

        # ── Tier ────────────────────────────────────────────────────
        tier = cfg.get("tier", {})
        self.tier_core_access_threshold = int(
            tier.get("coreAccessThreshold", 10))
        self.tier_core_composite_threshold = float(
            tier.get("coreCompositeThreshold", 0.7))
        self.tier_peripheral_composite_threshold = float(
            tier.get("peripheralCompositeThreshold", 0.15))
        self.tier_peripheral_age_days = int(
            tier.get("peripheralAgeDays", 60))
        self.tier_working_access_threshold = int(
            tier.get("workingAccessThreshold", 3))

        # ── Storage ────────────────────────────────────────────────
        self.db_path = cfg.get(
            "dbPath",
            str(get_hermes_home() / "lancedb_pro_data"))

        # ── Session Memory ──────────────────────────────────────────
        self.session_strategy = cfg.get("sessionStrategy", "none")

        # ── Management Tools ─────────────────────────────────────────
        self.enable_management_tools = bool(
            cfg.get("enableManagementTools", True))

        # ── Self Improvement ──────────────────────────────────────────
        si = cfg.get("selfImprovement", {})
        self.self_improvement_enabled = bool(si.get("enabled", False))
        self.self_improvement_before_reset = bool(si.get("beforeResetNote", False))
        self.self_improvement_skip_subagent = bool(si.get("skipSubagentBootstrap", False))
        self.self_improvement_ensure_files = bool(si.get("ensureLearningFiles", False))

    def save(self, hermes_home: Optional[Path] = None) -> None:
        """Write current config back to lancedb_pro.json."""
        hp = hermes_home or get_hermes_home()
        cfg_path = hp / "lancedb_pro.json"
        # Rebuild minimal config dict from current values
        data = {
            "embedding": {
                "provider": self.emb_provider,
                "apiKey": self.emb_api_key,
                "baseURL": self.emb_base_url,
                "model": self.emb_model,
                "dimensions": self.emb_dimensions,
                "normalized": self.emb_normalized,
                "taskQuery": self.emb_task_query,
                "taskPassage": self.emb_task_passage,
                "chunking": self.emb_chunking,
                "requestDimensions": self.emb_request_dimensions,
                "omitDimensions": self.emb_omit_dimensions,
            },
            "rerank": {
                "enabled": self.rerank_enabled,
                "provider": self.rerank_provider,
                "apiKey": self.rerank_api_key,
                "endpoint": self.rerank_endpoint,
                "model": self.rerank_model,
                "timeoutMs": self.rerank_timeout_ms,
            },
            "llm": {
                "apiKey": self.llm_api_key,
                "baseURL": self.llm_base_url,
                "model": self.llm_model,
                "timeoutMs": self.llm_timeout_ms,
            },
            "retrieval": {
                "mode": self.ret_mode,
                "vectorWeight": self.ret_vector_weight,
                "bm25Weight": self.ret_bm25_weight,
                "minScore": self.ret_min_score,
                "hardMinScore": self.ret_hard_min_score,
                "candidatePoolSize": self.ret_candidate_pool_size,
                "recencyHalfLifeDays": self.ret_recency_half_life_days,
                "recencyWeight": self.ret_recency_weight,
                "filterNoise": self.ret_filter_noise,
                "lengthNormAnchor": self.ret_length_norm_anchor,
                "timeDecayHalfLifeDays": self.ret_time_decay_half_life_days,
                "reinforcementFactor": self.ret_reinforcement_factor,
                "maxHalfLifeMultiplier": self.ret_max_half_life_multiplier,
            },
            "autoCapture": self.auto_capture,
            "autoRecall": self.auto_recall,
            "autoRecallMinLength": self.auto_recall_min_length,
            "autoRecallMinRepeated": self.auto_recall_min_repeated,
            "autoRecallMaxQueryLength": self.auto_recall_max_query_length,
            "autoRecallTimeoutMs": self.auto_recall_timeout_ms,
            "autoRecallPerItemMaxChars": self.auto_recall_per_item_max_chars,
            "autoRecallMaxChars": self.auto_recall_max_chars_total,
            "autoRecallExcludeAgents": self.recall_exclude_agents,
            "autoRecallIncludeAgents": self.recall_include_agents,
            "recallMode": self.recall_mode,
            "maxRecallPerTurn": self.max_recall_per_turn,
            "captureAssistant": self.capture_assistant,
            "smartExtraction": self.smart_extraction,
            "extractMinMessages": self.extract_min_messages,
            "extractMaxChars": self.extract_max_chars,
            "dbPath": self.db_path,
            "sessionStrategy": self.session_strategy,
            "enableManagementTools": self.enable_management_tools,
            "selfImprovement": {
                "enabled": self.self_improvement_enabled,
                "beforeResetNote": self.self_improvement_before_reset,
                "skipSubagentBootstrap": self.self_improvement_skip_subagent,
                "ensureLearningFiles": self.self_improvement_ensure_files,
            },
        }
        cfg_path.write_text(json.dumps(data, indent=2, ensure_ascii=False),
                            encoding="utf-8")
