# LanceDB-Pro Memory Provider

Python port of memory-lancedb-pro for Hermes Agent. Hybrid vector + BM25 retrieval with cross-encoder reranking, Weibull lifecycle decay, and automatic memory extraction.

## Config File

`$HERMES_HOME/lancedb_pro.json`:

```json
{
  "embedding": {
    "provider": "openai-compatible",
    "apiKey": "your-siliconflow-key",
    "baseURL": "https://api.siliconflow.cn/v1",
    "model": "BAAI/bge-m3",
    "dimensions": 1024
  },
  "rerank": {
    "enabled": true,
    "provider": "siliconflow",
    "apiKey": "your-siliconflow-key",
    "endpoint": "https://api.siliconflow.cn/v1/rerank",
    "model": "BAAI/bge-reranker-v2-m3"
  },
  "llm": {
    "apiKey": "your-minimax-key",
    "baseURL": "https://api.minimaxi.com/anthropic/v1",
    "model": "MiniMax-M2.7"
  },
  "retrieval": {
    "mode": "hybrid",
    "vectorWeight": 0.7,
    "bm25Weight": 0.3,
    "candidatePoolSize": 20,
    "hardMinScore": 0.35,
    "recencyHalfLifeDays": 14,
    "recencyWeight": 0.1,
    "timeDecayHalfLifeDays": 60,
    "reinforcementFactor": 0.5,
    "maxHalfLifeMultiplier": 3,
    "filterNoise": true,
    "lengthNormAnchor": 500
  },
  "autoCapture": true,
  "autoRecall": true,
  "captureAssistant": false,
  "smartExtraction": true,
  "extractMinMessages": 4,
  "extractMaxChars": 8000
}
```

## Tools

| Tool | Description |
|------|-------------|
| `memory_store` | Store an explicit memory fact |
| `memory_search` | Hybrid search with reranking |
| `memory_profile` | User profile facts |
| `memory_stats` | Store statistics and health |
| `memory_forget` | Delete a memory by ID |
| `memory_list` | List memories by scope/category |

## Setup

1. Install dependencies:
   ```bash
   pip install lancedb rank_bm25 httpx
   ```

2. Write config to `$HERMES_HOME/lancedb_pro.json`

3. Enable:
   ```bash
   hermes config set memory.provider lancedb-pro
   ```
