#!/usr/bin/env python3
"""
Score 5k instruction-response pairs with an LLM-as-a-judge on 4 axes and prune top-1k.

Inputs
- A JSONL/CSV file with columns: id (optional), instruction, response

Outputs
- scored.jsonl: each row + per-axis scores, rationale, aggregate
- top1k.jsonl: top 1,000 rows by aggregate (desc)

Usage
  python Prune.py \
    --in data.jsonl \
    --out_scored scored.jsonl \
    --out_top top1k.jsonl \
    --provider openai \
    --model gpt-4o-mini \
    --seed 42 \
    --max_concurrency 4

Environment
  For OpenAI: export OPENAI_API_KEY=...
  For Anthropic: export ANTHROPIC_API_KEY=...

Notes
- Uses JSON-constrained outputs for robust parsing.
- Retries with exponential backoff.
- Deterministic tie-breakers: higher aggregate, then higher min-axis score, then longer response.
- Provider adapters kept minimal; extend as needed.
"""
from __future__ import annotations
import argparse
import csv
import dataclasses as dc
import json
import random
import time
from typing import Any, Dict, Iterable, List, Tuple

try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **kwargs):
        return x

# --------------------------- Data structures ---------------------------
@dc.dataclass
class QAItem:
    id: str
    instruction: str
    response: str

@dc.dataclass
class Judged:
    item: QAItem
    helpfulness: int
    factuality: int
    completeness: int
    adherence: int
    rationale: str
    aggregate: float

# --------------------------- IO helpers ---------------------------
def read_any(path: str) -> List[QAItem]:
    items: List[QAItem] = []
    if path.endswith('.jsonl'):
        with open(path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                rec = json.loads(line)
                q = rec.get('instruction') or rec.get('prompt') or rec.get('input')
                a = rec.get('response') or rec.get('answer') or rec.get('output')
                if q is None or a is None:
                    continue
                _id = str(rec.get('id', i))
                items.append(QAItem(id=_id, instruction=str(q), response=str(a)))
    elif path.endswith('.csv'):
        with open(path, newline='', encoding='utf-8') as f:
            rdr = csv.DictReader(f)
            for i, rec in enumerate(rdr):
                q = rec.get('instruction') or rec.get('prompt') or rec.get('input')
                a = rec.get('response') or rec.get('answer') or rec.get('output')
                if q is None or a is None:
                    continue
                _id = str(rec.get('id', i))
                items.append(QAItem(id=_id, instruction=str(q), response=str(a)))
    else:
        raise ValueError('Unsupported file type; use .jsonl or .csv')
    return items

def write_jsonl(path: str, rows: Iterable[Dict[str, Any]]):
    with open(path, 'w', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

# --------------------------- Prompting ---------------------------
SYSTEM_PROMPT = (
    "You are a strict, detail-oriented evaluator for instruction-response pairs. "
    "Score the provided response to the instruction on four axes from 1 (poor) to 10 (excellent). "
    "Be consistent, avoid verbosity, and do not fix or rewrite the response. Just evaluate it. "
    "Assume general knowledge cutoff today unless the instruction explicitly requires real-time facts—if so, penalize factuality."
)

USER_TEMPLATE = (
    "Evaluate the following instruction-response. Provide integer scores 1-10 and a short rationale (<60 words).\n\n"
    "Instruction:\n{instruction}\n\nResponse:\n{response}\n\n"
    "Scoring criteria (1-10 each):\n"
    "1) Helpfulness: Does the response directly address the user's need and provide useful guidance?\n"
    "2) Factuality: Are the claims accurate and non-hallucinatory?\n"
    "3) Completeness: Does it cover the key parts of the instruction without major omissions?\n"
    "4) Adherence: Does it follow instructions (format, tone, constraints) and avoid unsafe content?\n\n"
    "Return ONLY a JSON object with fields: helpfulness, factuality, completeness, adherence, rationale."
)

JUDGE_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "helpfulness": {"type": "integer", "minimum": 1, "maximum": 10},
        "factuality": {"type": "integer", "minimum": 1, "maximum": 10},
        "completeness": {"type": "integer", "minimum": 1, "maximum": 10},
        "adherence": {"type": "integer", "minimum": 1, "maximum": 10},
        "rationale": {"type": "string"},
    },
    "required": ["helpfulness", "factuality", "completeness", "adherence", "rationale"],
    "additionalProperties": False,
}

# --------------------------- Utilities ---------------------------
def extract_json(s: str) -> str:
    start = s.find('{')
    end = s.rfind('}')
    if start == -1 or end == -1 or end < start:
        raise ValueError('No JSON object found in text')
    return s[start:end+1]

def safe_int(x: Any, lo=1, hi=10) -> int:
    try:
        v = int(x)
    except Exception:
        v = lo
    return max(lo, min(hi, v))

def aggregate_scores(helpfulness: int, factuality: int, completeness: int, adherence: int) -> float:
    return (helpfulness + factuality + completeness + adherence) / 4.0

# --------------------------- Main pipeline ---------------------------
def run(
    in_path: str,
    out_scored: str,
    out_top: str,
    provider: str,
    model: str,
    seed: int,
    max_concurrency: int,
):
    random.seed(seed)
    items = read_any(in_path)
    if not items:
        raise SystemExit('No items read from input. Check file and columns (instruction/response).')

    if provider == 'openai':
        judge: BaseJudge = OpenAIJudge(model)
    elif provider == 'anthropic':
        judge = AnthropicJudge(model)
    elif provider == 'mock':
        judge = MockJudge(seed)
    else:
        raise SystemExit(f'Unknown provider: {provider}')

    judged_rows: List[Judged] = []
    for it in tqdm(items, desc='Scoring', total=len(items)):
        data = judge.score(it.instruction, it.response)
        h = safe_int(data.get('helpfulness'))
        f = safe_int(data.get('factuality'))
        c = safe_int(data.get('completeness'))
        a = safe_int(data.get('adherence'))
        agg = aggregate_scores(h, f, c, a)
        judged_rows.append(Judged(item=it, helpfulness=h, factuality=f, completeness=c, adherence=a, rationale=str(data.get('rationale', '')), aggregate=agg))

    scored_out = []
    for r in judged_rows:
        scored_out.append({
            'id': r.item.id,
            'instruction': r.item.instruction,
            'response': r.item.response,
            'helpfulness': r.helpfulness,
            'factuality': r.factuality,
            'completeness': r.completeness,
            'adherence': r.adherence,
            'aggregate': round(r.aggregate, 3),
            'rationale': r.rationale,
        })
    write_jsonl(out_scored, scored_out)

    def tie_key(r: Judged) -> Tuple[float, int, int]:
        min_axis = min(r.helpfulness, r.factuality, r.completeness, r.adherence)
        resp_len = len(r.item.response)
        return (r.aggregate, min_axis, resp_len)

    topk = sorted(judged_rows, key=tie_key, reverse=True)[:1000]
    write_jsonl(out_top, ({
        'id': r.item.id,
        'instruction': r.item.instruction,
        'response': r.item.response,
        'helpfulness': r.helpfulness,
        'factuality': r.factuality,
        'completeness': r.completeness,
        'adherence': r.adherence,
        'aggregate': round(r.aggregate, 3),
        'rationale': r.rationale,
    } for r in topk))

    print(f"\nWrote {len(judged_rows)} scored rows to {out_scored}")
    print(f"Wrote top-1k to {out_top}")

# --------------------------- CLI ---------------------------
if __name__ == '__main__':
    p = argparse.ArgumentParser(description='LLM-as-a-judge scoring and pruning to top-1k')
    p.add_argument('--in', dest='in_path', required=True, help='Input file (.jsonl or .csv) with columns: instruction, response[, id]')
    p.add_argument('--out_scored', default='scored.jsonl', help='Output JSONL with per-axis scores + aggregate')
    p.add_argument('--out_top', default='top1k.jsonl', help='Output JSONL with top-1k by aggregate')
    p.add_argument('--provider', choices=['openai', 'anthropic', 'mock'], default='openai')
    p.add_argument('--model', default='gpt-4o-mini', help='Judge model name (e.g., OpenAI: gpt-4o-mini, Anthropic: claude-3-5-sonnet)')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--max_concurrency', type=int, default=4, help='Reserved for future async version; currently unused')
    args = p.parse_args()

    run(
        in_path=args.in_path,
        out_scored=args.out_scored,
        out_top=args.out_top,
        provider=args.provider,
        model=args.model,
        seed=args.seed,
        max_concurrency=args.max_concurrency,
    )

