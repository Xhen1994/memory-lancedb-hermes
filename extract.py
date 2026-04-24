"""Smart memory extraction for lancedb-pro.

LLM-powered 6-category classification from conversation turns.
Strips tool calls, code blocks, denials before extraction.

Also provides trigger-based capture (shouldCapture / detectCategory)
ported from the TypeScript reference project.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# 8 memory categories (mirrors memory-lancedb-pro)
CATEGORIES = [
    "profile",      # Who the user is (name, background, traits)
    "preference",   # User preferences and habits
    "entity",       # Entities mentioned (projects, tools, people)
    "decision",     # Decisions made or conclusions reached
    "event",       # Events that happened
    "case",        # Problem-solving cases or troubleshooting steps
    "pattern",     # Recurring patterns
    "reflection",  # Self-reflection
]

# ── Noise detection ────────────────────────────────────────────────────────

# Noise patterns — strings that indicate low-value content
NOISE_PATTERNS = [
    re.compile(r"^(hi|hello|hey|yo|good morning|good afternoon|good evening)[!.]?$", re.I),
    re.compile(r"^(thanks?|thank you|thx|ty|cheers)[!.]?$", re.I),
    re.compile(r"^(okay|ok|okk|sure|yes|no|nah|yep|nope|alright)[!.]?$", re.I),
    re.compile(r"^(sorry|apologies|my bad|my mistake)[!.]?$", re.I),
    re.compile(r"^(bye|goodbye|see you|have a good|have a nice)[!.]?$", re.I),
    re.compile(r"^(what('s| is) your name|who are you|tell me about yourself)[?.]?$", re.I),
    re.compile(r"^(can you|could you|would you|please help|help me)[?]$", re.I),
    re.compile(r"^[(]repeating previous instruction[)]$", re.I),
    re.compile(r"^(just|simply) (checking|confirming|asking)[.!?]$", re.I),
    re.compile(r"^nevermind[.,!?]?$", re.I),
    re.compile(r"^\[?(image|photo|picture|screenshot|attachment)\]?[:.]?$", re.I),
    re.compile(r"^/[\w]+$"),  # slash commands
]

# Minimal denial patterns
DENIAL_PATTERNS = [
    re.compile(r"i (don't|do not|am not|can('t| not)|won't|will not) know", re.I),
    re.compile(r"i ('m|am) not sure", re.I),
    re.compile(r"(i|we) (can't|cannot|could not) (tell|know|find|answer)", re.I),
    re.compile(r"no (idea|information|nothing) (is )?(available|known)", re.I),
    re.compile(r"(i|we) (don('t| not)|do not) have (that|any) information", re.I),
]


def is_noise(text: str) -> bool:
    """Check if a text string is likely noise / low-value."""
    if not text or len(text.strip()) < 8:
        return True
    stripped = text.strip()
    for pat in NOISE_PATTERNS:
        if pat.fullmatch(stripped):
            return True
    # Very short single-word lines (English-centric: relies on whitespace split)
    # For Chinese, use char count only to avoid false positives
    if len(stripped.split()) <= 2 and len(stripped) <= 20:
        # If text has CJK chars, rely on absolute length only
        import re
        if re.search(r'[\u4e00-\u9fff]', stripped):
            if len(stripped) < 10:
                return True
        else:
            return True
    return False


def is_denial(text: str) -> bool:
    """Check if text is an agent denial / non-answer."""
    for pat in DENIAL_PATTERNS:
        if pat.search(text):
            return True
    return False


# ── Trigger-based capture (ported from TypeScript) ────────────────────────

# Memory triggers — keywords and patterns that indicate memorizable content
MEMORY_TRIGGERS = [
    re.compile(r"\b(we )?decided\b|we'?ll use|we will use|switch(ed)? to|migrate(d)? to|going forward|from now on", re.I),
    re.compile(r"\b(remember|zapamatuj si|pamatuj)\b", re.I),
    re.compile(r"\b(prefer|preferuji|radši|nechci)\b", re.I),
    re.compile(r"\+\d{10,}"),  # phone numbers
    re.compile(r"[\w.-]+@[\w.-]+\.\w+"),  # email addresses
    re.compile(r"my\s+\w+\s+is|is\s+my", re.I),
    re.compile(r"i (like|prefer|hate|love|want|need|care)\b", re.I),
    re.compile(r"\b(always|never|important)\b", re.I),
    # Chinese triggers (Simplified + Traditional)
    re.compile(r"记住|記住|记一下|記一下|别忘了|別忘了|备注|備註"),
    re.compile(r"偏好|喜好|喜欢|喜歡|讨厌|討厭|不喜欢|不喜歡|爱用|愛用|习惯|習慣"),
    re.compile(r"决定|決定|选择了|選擇了|改用|换成|換成|以后用|以後用"),
    re.compile(r"我的\S+是|叫我|称呼|稱呼"),
    re.compile(r"总是|總是|从不|從不|一直|每次都|老是"),
    re.compile(r"重要|关键|關鍵|注意|千万别|千萬別"),
    re.compile(r"帮我|筆記|存档|存起来|存一下|重点|原则|底线"),
]

# Patterns that should NOT be captured (memory management meta-ops)
CAPTURE_EXCLUDE_PATTERNS = [
    re.compile(r"\b(memory-pro|memory_store|memory_recall|memory_forget|memory_update)\b", re.I),
    re.compile(r"\b(delete|remove|forget|purge|cleanup|clean up|clear)\b.*\b(memory|memories|entry|entries)\b", re.I),
    re.compile(r"\b(memory|memories)\b.*\b(delete|remove|forget|purge|cleanup|clean up|clear)\b", re.I),
    re.compile(r"\bhow do i\b.*\b(delete|remove|forget|purge|cleanup|clear)\b", re.I),
    re.compile(r"(删除|刪除|清理|清除).{0,12}(记忆|記憶|memory)", re.I),
]


def should_capture(text: str) -> bool:
    """Check if a user message contains memorizable content.

    Uses keyword-based triggers — fast, no LLM needed.
    Ported from TypeScript shouldCapture().
    """
    s = text.strip()

    # CJK characters carry more meaning per character
    has_cjk = bool(re.search(r'[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff\uac00-\ud7af]', s))
    min_len = 4 if has_cjk else 10
    if len(s) < min_len or len(s) > 500:
        return False

    # Skip injected context from memory recall
    if "<relevant-memories>" in s:
        return False
    # Skip system-generated content
    if s.startswith("<") and "</" in s:
        return False
    # Skip agent summary responses
    if "**" in s and "\n-" in s:
        return False
    # Skip emoji-heavy responses (likely agent output)
    emoji_count = len(re.findall(r'[\U0001F300-\U0001F9FF]', s))
    if emoji_count > 3:
        return False

    # Exclude memory management prompts
    if any(r.search(s) for r in CAPTURE_EXCLUDE_PATTERNS):
        return False

    return any(r.search(s) for r in MEMORY_TRIGGERS)


def detect_category(text: str) -> str:
    """Detect memory category from text content using keyword heuristics.

    Returns one of: preference, decision, entity, profile, other.
    Ported from TypeScript detectCategory().
    """
    lower = text.lower()

    if re.search(r"prefer|like|love|hate|want|偏好|喜欢|喜歡|讨厌|討厭|不喜欢|不喜歡|爱用|愛用|习惯|習慣", lower, re.I):
        return "preference"
    if re.search(r"decided|we decided|will use|we will use|we'll use|switch(ed)? to|migrate(d)? to|going forward|from now on|决定|決定|选择了|選擇了|改用|换成|換成|以后用|以後用", lower, re.I):
        return "decision"
    if re.search(r"\+\d{10,}|@[\w.-]+\.\w+|is called|我的\S+是|叫我|称呼|稱呼", lower, re.I):
        return "entity"
    if re.search(r"my\s+\w+\s+is|i am|i'm|我是|名字|name", lower, re.I):
        return "profile"
    if re.search(r"\b(is|are|has|have)\b|总是|從不|一直|每次都|老是", lower, re.I):
        return "preference"

    return "other"


# ── Text cleaning ──────────────────────────────────────────────────────────

def strip_tool_calls(text: str) -> str:
    """Remove tool call blocks from text before extraction."""
    # Remove markdown code blocks
    text = re.sub(r"```[\s\S]*?```", "", text)
    # Remove inline code
    text = re.sub(r"`[^`]+`", "", text)
    # Remove tool result markers
    text = re.sub(r"<tool_result>[\s\S]*?</tool_result>", "", text, flags=re.I)
    text = re.sub(r"\\[\\[.*?\\]\\]", "", text)
    # Remove URLs
    text = re.sub(r"https?://\S+", "", text)
    # Collapse whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def clean_for_extraction(text: str) -> str:
    """Apply all cleaning steps for extraction input."""
    text = strip_tool_calls(text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ── Smart Extractor ────────────────────────────────────────────────────────

class SmartExtractor:
    """LLM-powered memory extractor.

    Takes conversation turns and extracts structured memories using an LLM.
    Falls back to rule-based extraction if LLM is unavailable.
    """

    def __init__(
        self,
        llm_api_key: str,
        llm_base_url: str = "https://api.minimaxi.com/anthropic/v1",
        llm_model: str = "MiniMax-M2.7",
        timeout_ms: int = 60000,
    ):
        self.llm_api_key = llm_api_key
        self.llm_base_url = llm_base_url.rstrip("/")
        self.llm_model = llm_model
        self.timeout = timeout_ms / 1000.0
        self._client = None

    async def _get_client(self):
        if self._client is None:
            import httpx
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.timeout),
                headers={
                    "Authorization": f"Bearer {self.llm_api_key}",
                    "Content-Type": "application/json",
                    "anthropic-version": "2023-06-01",
                },
            )
        return self._client

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    def extract_memories_sync(
        self,
        turns: List[Dict[str, str]],
        max_chars: int = 8000,
    ) -> List[Dict[str, Any]]:
        """Synchronous wrapper for extract_memories."""
        import asyncio
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                return ex.submit(
                    asyncio.run,
                    self.extract_memories(turns, max_chars)
                ).result(timeout=self.timeout + 5)
        else:
            return asyncio.run(self.extract_memories(turns, max_chars))

    async def extract_memories(
        self,
        turns: List[Dict[str, str]],
        max_chars: int = 8000,
    ) -> List[Dict[str, Any]]:
        """Extract structured memories from conversation turns.

        Each turn: {"role": "user"|"assistant", "content": str}

        Returns list of:
          {"text": str, "category": str, "importance": float, "confidence": float}
        """
        if not turns:
            return []

        # Build the extraction prompt
        prompt = self._build_prompt(turns, max_chars)
        if not prompt:
            return []

        try:
            return await self._llm_extract(prompt)
        except Exception as e:
            logger.warning("LLM extraction failed, using rule fallback: %s", e)
            return self._rule_extract(turns)

    def _build_prompt(
        self,
        turns: List[Dict[str, str]],
        max_chars: int,
    ) -> str:
        """Build the extraction prompt from conversation turns."""
        role_map = {"user": "User", "assistant": "Assistant", "system": "System"}
        lines = []
        total_chars = 0
        max_chars = int(max_chars * 0.8)  # leave headroom

        for turn in reversed(turns):
            role = role_map.get(turn.get("role", ""), "Other")
            content = turn.get("content", "")
            content = clean_for_extraction(content)

            if is_noise(content) or is_denial(content):
                continue

            if not content or len(content) < 10:
                continue

            turn_text = f"{role}: {content}"
            if total_chars + len(turn_text) > max_chars:
                break
            lines.append(turn_text)
            total_chars += len(turn_text)

        lines.reverse()
        if not lines:
            return ""

        conversation = "\n\n".join(lines)
        categories_str = ", ".join(CATEGORIES[:-1]) + f", {CATEGORIES[-1]}"

        prompt = f"""You are a memory extraction system. From the conversation below, extract all meaningful facts, preferences, decisions, and patterns that an AI assistant should remember long-term.

For each memory, provide:
1. The memory text (concise, factual, 20-150 chars)
2. The category: {categories_str}
3. A confidence score (0.0-1.0) for how certain this memory is extractable

Rules:
- Only extract information about the USER, not the assistant
- Skip denials, meta-questions, apologies, generic greetings
- Combine related facts into single memories when possible
- Return ONLY valid JSON array, no explanation

Categories:
- profile: user identity, name, background, traits
- preference: preferences, habits, communication style
- entity: projects, tools, companies, people mentioned
- decision: conclusions, agreements, plans made
- event: things that happened
- case: problems solved, how-to knowledge
- pattern: recurring behaviors or tendencies
- reflection: self-improvement notes

Conversation:
{conversation}

Output JSON array:
[{{"text": "...", "category": "...", "importance": 0.8, "confidence": 0.9}}]"""
        return prompt

    async def _llm_extract(self, prompt: str) -> List[Dict[str, Any]]:
        """Call LLM for memory extraction."""
        client = await self._get_client()
        url = f"{self.llm_base_url}/messages"

        body = {
            "model": self.llm_model,
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": prompt}],
        }

        resp = await client.post(url, json=body)
        resp.raise_for_status()
        data = resp.json()

        content = data.get("content", [])
        if isinstance(content, list) and content:
            text = content[0].get("text", "")
        else:
            text = str(content)

        # Extract JSON from response
        json_match = re.search(
            r"\[\s*\{[\s\S]*\}\s*\]", text, re.MULTILINE)
        if json_match:
            try:
                items = json.loads(json_match.group())
                return [dict(item) for item in items]
            except Exception:
                pass

        logger.warning("Could not parse LLM extraction response: %s", text[:200])
        return []

    def _rule_extract(
        self,
        turns: List[Dict[str, str]],
    ) -> List[Dict[str, Any]]:
        """Rule-based extraction fallback.

        Looks for explicit preference/decision patterns.
        """
        results = []

        preference_patterns = [
            re.compile(r"\bi\s+(?:prefer|like|love|hate|always|never|usually)\s+(.{10,100})", re.I),
            re.compile(r"\bmy\s+(?:favorite|preferred|default)\s+\w+\s+is\s+(.{5,80})", re.I),
            re.compile(r"\b(i'm|i am)\s+(?:a\s+)?(?:an?\s+)?(.+?)(?:\.|,|\s{2})", re.I),
        ]

        decision_patterns = [
            re.compile(r"\bwe\s+(?:decided|agreed|chose|settled on)\s+(?:to\s+)?(.+?)(?:\.|,|\s{2})", re.I),
            re.compile(r"\b(?:let's|let us)\s+(?:use|go with|pick|choose)\s+(.+?)(?:\.|,|\s{2})", re.I),
        ]

        for turn in turns:
            if turn.get("role") != "user":
                continue
            content = clean_for_extraction(turn.get("content", ""))
            if not content or len(content) < 15:
                continue

            for pat in preference_patterns:
                m = pat.search(content)
                if m:
                    results.append({
                        "text": m.group(0)[:200],
                        "category": "preference",
                        "importance": 0.7,
                        "confidence": 0.6,
                    })

            for pat in decision_patterns:
                m = pat.search(content)
                if m:
                    results.append({
                        "text": m.group(0)[:200],
                        "category": "decision",
                        "importance": 0.8,
                        "confidence": 0.7,
                    })

        return results
