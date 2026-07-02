import argparse
import html
import json
import math
import os
import re
from typing import Iterable


REGION_LABELS = {
    "auth": "AUTH",
    "data": "DATA",
    "data_fact": "FACT",
    "data_attack": "ATTACK",
    "special": "SPECIAL",
}

REGION_CLASSES = {
    "auth": "region-auth",
    "data": "region-data",
    "data_fact": "region-fact",
    "data_attack": "region-attack",
    "special": "region-special",
}

FILTER_REGIONS = ("auth", "data", "data_fact", "data_attack")
CONTROL_TOKENS = {
    "system",
    "user",
    "data",
    "data_fact",
    "data_attack",
    "im_start",
    "im_end",
}
CONTROL_FRAGMENTS = {"system", "user", "data", "fact", "attack"}


def build_data(data_fact: str, data_attack: str) -> str:
    return (
        "<data>\n"
        "<data_fact>\n"
        f"{data_fact}\n"
        "</data_fact>\n\n"
        "<data_attack>\n"
        f"{data_attack}\n"
        "</data_attack>\n"
        "</data>\n"
    )


def clean_attention_token(token: str) -> str:
    return (
        token.replace("Ġ", " ")
        .replace("▁", " ")
        .replace("Ċ", "\n")
        .replace("<|", "")
        .replace("|>", "")
        .strip()
    )


def is_meaningful_attention_token(token: str) -> bool:
    cleaned = clean_attention_token(token)
    if not cleaned:
        return False
    normalized = cleaned.strip("<>/[](){}:;,.!?\"'` \t\r\n").lower()
    if not normalized or normalized in CONTROL_TOKENS:
        return False
    if normalized.startswith("_") and normalized.strip("_-") in CONTROL_FRAGMENTS:
        return False
    return bool(re.search(r"[A-Za-z0-9\u4e00-\u9fff]", normalized))


def read_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str, rows: list[dict]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def select_record(rows: list[dict], record_id: str | None, index: int) -> dict:
    if record_id is not None:
        for row in rows:
            if str(row.get("id")) == record_id:
                return row
        raise ValueError(f"No attention record found with id={record_id!r}.")
    if index < 0 or index >= len(rows):
        raise IndexError(f"--index {index} is out of range for {len(rows)} records.")
    return rows[index]


def result_index(rows: list[dict]) -> dict[tuple[str, str], dict]:
    indexed = {}
    for row in rows:
        indexed[(str(row.get("dataset", "")), str(row.get("id", "")))] = row
    return indexed


def text_range(text: str, needle: str, start: int = 0) -> tuple[int, int] | None:
    if not needle:
        return None
    pos = text.find(needle, start)
    if pos < 0:
        return None
    return pos, pos + len(needle)


def token_span_from_offsets(tokenizer, text: str, char_start: int, char_end: int) -> tuple[int, int] | None:
    if char_start < 0 or char_end <= char_start:
        return None
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    offsets = encoded.get("offset_mapping")
    if not offsets:
        return None
    indices = [
        i
        for i, (start, end) in enumerate(offsets)
        if end > char_start and start < char_end
    ]
    if not indices:
        return None
    return indices[0], indices[-1] + 1


def repaired_ranges(record: dict, result: dict, tokenizer) -> dict:
    ranges = dict(record.get("token_ranges") or {})
    sample = result.get("sample") or {}
    data_range = ranges.get("data")
    if not data_range:
        return ranges
    data_start = int(data_range[0])
    data_text = build_data(sample.get("data_fact", ""), sample.get("data_attack", ""))
    fact_range = text_range(data_text, sample.get("data_fact", ""))
    attack_text = sample.get("data_attack", "")
    attack_range = text_range(data_text, attack_text) if attack_text else None
    if fact_range:
        span = token_span_from_offsets(tokenizer, data_text, *fact_range)
        if span:
            ranges["data_fact"] = [data_start + span[0], data_start + span[1]]
    if attack_range:
        span = token_span_from_offsets(tokenizer, data_text, *attack_range)
        if span:
            ranges["data_attack"] = [data_start + span[0], data_start + span[1]]
    return ranges


def region_for_index(index: int, ranges: dict) -> str:
    for name in ("data_attack", "data_fact", "auth", "data"):
        if name in ranges:
            start, end = ranges[name]
            if int(start) <= index < int(end):
                return name
    return "special"


def relabel_record_regions(record: dict, result: dict | None, tokenizer) -> dict:
    if not result or tokenizer is None:
        return record
    record = dict(record)
    ranges = repaired_ranges(record, result, tokenizer)
    tokens = []
    region_scores = {}
    for token in record.get("tokens", []):
        token = dict(token)
        region = region_for_index(int(token.get("i", -1)), ranges)
        if region not in FILTER_REGIONS:
            continue
        token["r"] = region
        score = float(token.get("s", 0.0))
        region_scores[region] = region_scores.get(region, 0.0) + score
        tokens.append(token)
    top_k = max(0, min(len(record.get("top_tokens", [])) or 80, len(tokens)))
    record["token_ranges"] = ranges
    record["tokens"] = tokens
    record["region_scores"] = region_scores
    record["top_tokens"] = sorted(tokens, key=lambda token: float(token.get("s", 0.0)), reverse=True)[:top_k]
    return record


def merge_result_metadata(record: dict, result_rows: list[dict] | None, tokenizer=None) -> dict:
    if not result_rows:
        return record
    result = result_index(result_rows).get((str(record.get("dataset", "")), str(record.get("id", ""))))
    if not result:
        return record
    record = relabel_record_regions(record, result, tokenizer)
    record = dict(record)
    attention = result.get("attention") or {}
    record["result"] = {
        "attention_shift": result.get("attention_shift"),
        "injection_status": result.get("injection_status"),
        "candidate_attention_shift_attack_failed": result.get("candidate_attention_shift_attack_failed"),
        "focus_score": attention.get("focus_score"),
        "threshold": attention.get("threshold"),
        "output": result.get("output"),
        "judge": result.get("judge"),
    }
    return record


def drop_special_attention(record: dict) -> dict:
    record = dict(record)
    tokens = [
        token
        for token in record.get("tokens", [])
        if token.get("r") in FILTER_REGIONS and is_meaningful_attention_token(str(token.get("t", "")))
    ]
    region_scores = {}
    for token in tokens:
        region = token.get("r")
        region_scores[region] = region_scores.get(region, 0.0) + float(token.get("s", 0.0))
    top_k = max(0, min(len(record.get("top_tokens", [])) or 80, len(tokens)))
    top_tokens = sorted(tokens, key=lambda token: float(token.get("s", 0.0)), reverse=True)[:top_k]
    record["tokens"] = tokens
    record["top_tokens"] = top_tokens
    record["region_scores"] = region_scores
    record["num_display_tokens"] = len(tokens)
    record["num_filtered_tokens"] = max(0, int(record.get("num_input_tokens", len(tokens))) - len(tokens))
    return record


def short_token(token: str) -> str:
    token = token.replace("Ġ", " ").replace("▁", " ")
    token = token.replace("\n", "\\n").replace("\t", "\\t")
    return token


def score_scale(tokens: list[dict]) -> tuple[float, float]:
    scores = sorted(float(token.get("s", 0.0)) for token in tokens)
    if not scores:
        return 0.0, 1.0
    lo = scores[0]
    hi_index = max(0, min(len(scores) - 1, int(math.ceil(len(scores) * 0.98)) - 1))
    hi = scores[hi_index]
    if hi <= lo:
        hi = scores[-1] if scores[-1] > lo else lo + 1e-12
    return lo, hi


def attention_alpha(score: float, lo: float, hi: float) -> float:
    value = (score - lo) / max(hi - lo, 1e-12)
    value = max(0.0, min(1.0, value))
    return 0.08 + value * 0.82


def render_tokens(tokens: list[dict]) -> str:
    lo, hi = score_scale(tokens)
    spans = []
    for token in tokens:
        score = float(token.get("s", 0.0))
        region = str(token.get("r", "special"))
        cls = REGION_CLASSES.get(region, "region-special")
        alpha = attention_alpha(score, lo, hi)
        title = (
            f"index: {token.get('i')}&#10;"
            f"region: {html.escape(region)}&#10;"
            f"score: {score:.8f}"
        )
        text = html.escape(short_token(str(token.get("t", "")))) or "&nbsp;"
        spans.append(
            f'<span class="token {cls}" data-region="{html.escape(region)}" data-score="{score:.12f}" '
            f'style="--attn-alpha:{alpha:.4f}" title="{title}">'
            f"{text}</span>"
        )
    return "\n".join(spans)


def status_class(value) -> str:
    text = str(value).lower()
    if text in ("true", "success"):
        return "status-bad"
    if text in ("false", "failed"):
        return "status-good"
    if text in ("ambiguous", "skipped", "unjudged", "none"):
        return "status-warn"
    return "status-neutral"


def bool_label(value) -> str:
    if value is True:
        return "YES"
    if value is False:
        return "NO"
    return "UNKNOWN"


def render_status_cards(record: dict) -> str:
    result = record.get("result") or {}
    attention_shift = result.get("attention_shift")
    injection_status = result.get("injection_status", "unknown")
    focus_score = result.get("focus_score")
    threshold = result.get("threshold")
    output = result.get("output")
    focus_text = "unknown" if focus_score is None else f"{float(focus_score):.6f}"
    threshold_text = "unknown" if threshold is None else f"{float(threshold):.6f}"
    output_html = ""
    if output:
        output_html = f'<div class="output"><strong>Output</strong><br>{html.escape(str(output))}</div>'
    return f"""
      <div class="status-grid">
        <div class="status-card">
          <span class="status-label">Attention Shift</span>
          <strong class="{status_class(attention_shift)}">{bool_label(attention_shift)}</strong>
          <small>focus {focus_text} / threshold {threshold_text}</small>
        </div>
        <div class="status-card">
          <span class="status-label">Attack Status</span>
          <strong class="{status_class(injection_status)}">{html.escape(str(injection_status).upper())}</strong>
          <small>from result JSONL when provided</small>
        </div>
      </div>
      {output_html}
    """


def render_filters(tokens: list[dict]) -> str:
    max_score = max((float(token.get("s", 0.0)) for token in tokens), default=0.0)
    controls = []
    for region in FILTER_REGIONS:
        controls.append(
            f"""
            <label class="check">
              <input type="checkbox" class="region-filter" value="{region}" checked>
              <span class="pill {REGION_CLASSES[region]}">{REGION_LABELS[region]}</span>
            </label>
            """
        )
    return f"""
      <section class="filters" aria-label="Token filters">
        <div class="filter-row">
          <div>
            <h2>Filters</h2>
            <div class="checks">{''.join(controls)}</div>
          </div>
          <div class="score-filter">
            <label for="min-score">Minimum score</label>
            <input id="min-score" type="number" min="0" max="{max_score:.12f}" step="0.000001" value="0">
            <input id="score-slider" type="range" min="0" max="{max_score:.12f}" step="0.000001" value="0">
          </div>
        </div>
        <div class="filter-actions">
          <button type="button" id="select-main-regions">AUTH/DATA/FACT/ATTACK</button>
          <button type="button" id="clear-regions">Clear regions</button>
          <button type="button" id="reset-filters">Reset</button>
          <span id="visible-count"></span>
        </div>
      </section>
    """


def render_legend() -> str:
    items = []
    for region in FILTER_REGIONS:
        items.append(
            f'<span class="legend-item"><span class="legend-swatch {REGION_CLASSES[region]}"></span>'
            f'{REGION_LABELS[region]}</span>'
        )
    return f'<div class="legend">{"".join(items)}<span class="legend-note">same fill scale; stronger fill = higher score</span></div>'


def render_region_scores(region_scores: dict) -> str:
    total = sum(float(v) for v in region_scores.values()) or 1.0
    items = []
    for region, score in sorted(region_scores.items(), key=lambda item: float(item[1]), reverse=True):
        score = float(score)
        pct = score / total * 100
        label = REGION_LABELS.get(region, region.upper())
        cls = REGION_CLASSES.get(region, "region-special")
        items.append(
            f"""
            <div class="metric">
              <div class="metric-head">
                <span class="pill {cls}">{html.escape(label)}</span>
                <span>{score:.6f}</span>
              </div>
              <div class="bar"><span style="width:{pct:.2f}%"></span></div>
              <div class="metric-foot">{pct:.2f}% of recorded token attention</div>
            </div>
            """
        )
    return "\n".join(items)


def render_top_tokens(tokens: Iterable[dict]) -> str:
    rows = []
    for token in tokens:
        region = str(token.get("r", "special"))
        cls = REGION_CLASSES.get(region, "region-special")
        rows.append(
            f"""
            <tr>
              <td>{token.get("i")}</td>
              <td><code>{html.escape(short_token(str(token.get("t", ""))))}</code></td>
              <td><span class="pill {cls}">{html.escape(REGION_LABELS.get(region, region.upper()))}</span></td>
              <td class="num">{float(token.get("s", 0.0)):.8f}</td>
            </tr>
            """
        )
    return "\n".join(rows)


def render_ranges(ranges: dict) -> str:
    rows = []
    for name, value in ranges.items():
        start, end = value
        cls = REGION_CLASSES.get(name, "region-special")
        rows.append(
            f"""
            <tr>
              <td><span class="pill {cls}">{html.escape(REGION_LABELS.get(name, name.upper()))}</span></td>
              <td class="num">{start}</td>
              <td class="num">{end}</td>
              <td class="num">{max(0, end - start)}</td>
            </tr>
            """
        )
    return "\n".join(rows)


def merge_all_result_metadata(records: list[dict], result_rows: list[dict] | None, tokenizer=None) -> list[dict]:
    if not result_rows:
        return records
    indexed = result_index(result_rows)
    merged = []
    for record in records:
        result = indexed.get((str(record.get("dataset", "")), str(record.get("id", ""))))
        if not result:
            merged.append(record)
            continue
        record = relabel_record_regions(record, result, tokenizer)
        record = dict(record)
        attention = result.get("attention") or {}
        record["result"] = {
            "attention_shift": result.get("attention_shift"),
            "injection_status": result.get("injection_status"),
            "candidate_attention_shift_attack_failed": result.get("candidate_attention_shift_attack_failed"),
            "focus_score": attention.get("focus_score"),
            "threshold": attention.get("threshold"),
            "output": result.get("output"),
            "judge": result.get("judge"),
        }
        merged.append(record)
    return merged


def render_all_filters(records: list[dict]) -> str:
    max_score = max(
        (float(token.get("s", 0.0)) for record in records for token in record.get("tokens", [])),
        default=0.0,
    )
    controls = []
    for region in FILTER_REGIONS:
        controls.append(
            f"""
            <label class="check">
              <input type="checkbox" class="region-filter" value="{region}" checked>
              <span class="pill {REGION_CLASSES[region]}">{REGION_LABELS[region]}</span>
            </label>
            """
        )
    return f"""
      <section class="filters" aria-label="Token filters">
        <div class="filter-row">
          <div>
            <h2>Filters</h2>
            <div class="checks">{''.join(controls)}</div>
          </div>
          <div class="score-filter">
            <label for="min-score">Minimum score</label>
            <input id="min-score" type="number" min="0" max="{max_score:.12f}" step="0.000001" value="0">
            <input id="score-slider" type="range" min="0" max="{max_score:.12f}" step="0.000001" value="0">
          </div>
        </div>
        <div class="filter-actions">
          <button type="button" id="select-main-regions">AUTH/DATA/FACT/ATTACK</button>
          <button type="button" id="clear-regions">Clear regions</button>
          <button type="button" id="reset-filters">Reset</button>
          <span id="visible-count"></span>
        </div>
      </section>
    """


def render_all_summary(records: list[dict]) -> str:
    total = len(records)
    shift = sum(1 for record in records if (record.get("result") or {}).get("attention_shift") is True)
    success = sum(1 for record in records if (record.get("result") or {}).get("injection_status") == "success")
    failed = sum(1 for record in records if (record.get("result") or {}).get("injection_status") == "failed")
    candidates = sum(
        1 for record in records if (record.get("result") or {}).get("candidate_attention_shift_attack_failed") is True
    )
    tokens = sum(len(record.get("tokens", [])) for record in records)
    return f"""
      <div class="summary-grid">
        <div class="summary-card"><span>Total Samples</span><strong>{total}</strong></div>
        <div class="summary-card"><span>Display Tokens</span><strong>{tokens}</strong></div>
        <div class="summary-card"><span>Attention Shift</span><strong>{shift}</strong></div>
        <div class="summary-card"><span>Attack Success</span><strong>{success}</strong></div>
        <div class="summary-card"><span>Attack Failed</span><strong>{failed}</strong></div>
        <div class="summary-card"><span>Shift + Failed</span><strong>{candidates}</strong></div>
      </div>
    """


def render_record_header(record: dict, index: int) -> str:
    result = record.get("result") or {}
    attention_shift = result.get("attention_shift")
    injection_status = result.get("injection_status", "unknown")
    focus_score = result.get("focus_score")
    threshold = result.get("threshold")
    focus_text = "unknown" if focus_score is None else f"{float(focus_score):.6f}"
    threshold_text = "unknown" if threshold is None else f"{float(threshold):.6f}"
    candidate = result.get("candidate_attention_shift_attack_failed")
    return f"""
      <summary>
        <span class="sample-index">#{index}</span>
        <span class="sample-id">{html.escape(str(record.get("dataset", "")))} / {html.escape(str(record.get("id", "")))}</span>
        <span class="badge {status_class(attention_shift)}">Shift {bool_label(attention_shift)}</span>
        <span class="badge {status_class(injection_status)}">Attack {html.escape(str(injection_status).upper())}</span>
        <span class="badge status-neutral">Candidate {bool_label(candidate)}</span>
        <span class="badge status-neutral">Focus {focus_text} / {threshold_text}</span>
        <span class="badge status-neutral">Tokens {record.get("num_display_tokens", len(record.get("tokens", [])))}</span>
      </summary>
    """


def render_compact_region_scores(region_scores: dict) -> str:
    if not region_scores:
        return ""
    total = sum(float(v) for v in region_scores.values()) or 1.0
    items = []
    for region, score in sorted(region_scores.items(), key=lambda item: float(item[1]), reverse=True):
        score = float(score)
        pct = score / total * 100
        items.append(
            f'<span class="region-score"><span class="pill {REGION_CLASSES.get(region, "region-special")}">'
            f'{html.escape(REGION_LABELS.get(region, region.upper()))}</span> {score:.6f} ({pct:.1f}%)</span>'
        )
    return "".join(items)


def render_all_record(record: dict, index: int) -> str:
    result = record.get("result") or {}
    output = result.get("output")
    output_html = ""
    if output:
        output_html = f'<div class="output"><strong>Output</strong><br>{html.escape(str(output))}</div>'
    open_attr = " open" if result.get("attention_shift") is True else ""
    return f"""
      <details class="sample"{open_attr} data-dataset="{html.escape(str(record.get("dataset", "")))}"
        data-attention-shift="{html.escape(str(result.get("attention_shift", "unknown")).lower())}"
        data-injection-status="{html.escape(str(result.get("injection_status", "unknown")).lower())}">
        {render_record_header(record, index)}
        <div class="sample-body">
          {output_html}
          <div class="record-meta">
            <span>generated token <code>{html.escape(str(record.get("generated_token", "")))}</code></span>
            <span>filtered tokens <code>{record.get("num_filtered_tokens", 0)}</code></span>
            {render_compact_region_scores(record.get("region_scores", {}))}
          </div>
          <div class="token-wall">
            {render_tokens(record.get("tokens", []))}
          </div>
        </div>
      </details>
    """


def render_all_html(records: list[dict], source_path: str) -> str:
    sample_html = "\n".join(render_all_record(record, index) for index, record in enumerate(records))
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Attention Tokens - All Samples</title>
  <style>
    :root {{
      color-scheme: light;
      --text: #1d2733;
      --muted: #667085;
      --line: #d9e0e8;
      --panel: #f7f9fb;
      --heat: 217, 72, 34;
      --auth: #2f6fbd;
      --data: #607080;
      --fact: #21845b;
      --attack: #b42318;
      --special: #7a5c00;
      --good: #177245;
      --bad: #b42318;
      --warn: #9a6700;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: var(--text);
      background: #ffffff;
      line-height: 1.45;
    }}
    header {{
      position: sticky;
      top: 0;
      z-index: 10;
      border-bottom: 1px solid var(--line);
      background: rgba(255, 255, 255, 0.96);
      padding: 18px 24px 14px;
    }}
    main {{
      padding: 18px 24px 32px;
    }}
    h1 {{
      font-size: 22px;
      line-height: 1.2;
      margin: 0 0 8px;
      overflow-wrap: anywhere;
    }}
    h2 {{
      font-size: 14px;
      margin: 0 0 8px;
      text-transform: uppercase;
      color: var(--muted);
      letter-spacing: 0;
    }}
    .meta, .record-meta {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px 12px;
      color: var(--muted);
      font-size: 13px;
    }}
    .meta code, .record-meta code {{
      background: #eef2f6;
      border: 1px solid var(--line);
      border-radius: 4px;
      padding: 2px 5px;
      color: var(--text);
    }}
    .summary-grid {{
      display: grid;
      grid-template-columns: repeat(6, minmax(0, 1fr));
      gap: 10px;
      margin: 14px 0;
    }}
    .summary-card {{
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px;
      background: #fff;
      min-width: 0;
    }}
    .summary-card span {{
      color: var(--muted);
      display: block;
      font-size: 12px;
    }}
    .summary-card strong {{
      display: block;
      font-size: 20px;
      line-height: 1.2;
      margin-top: 3px;
    }}
    .filters {{
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
      padding: 12px;
      margin: 14px 0 0;
    }}
    .filter-row {{
      display: grid;
      grid-template-columns: minmax(0, 1fr) 280px;
      gap: 16px;
      align-items: end;
    }}
    .checks, .filter-actions {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      align-items: center;
    }}
    .filter-actions {{ margin-top: 10px; }}
    .check {{
      display: inline-flex;
      align-items: center;
      gap: 5px;
      cursor: pointer;
    }}
    .score-filter {{
      display: grid;
      gap: 6px;
      font-size: 12px;
      color: var(--muted);
    }}
    input[type="number"], input[type="range"] {{ width: 100%; }}
    input[type="number"] {{
      border: 1px solid var(--line);
      border-radius: 5px;
      padding: 6px 7px;
      font: inherit;
      font-variant-numeric: tabular-nums;
    }}
    button {{
      border: 1px solid var(--line);
      border-radius: 5px;
      background: #f7f9fb;
      color: var(--text);
      padding: 6px 9px;
      font: inherit;
      font-size: 12px;
      cursor: pointer;
    }}
    #visible-count {{
      color: var(--muted);
      font-size: 12px;
      margin-left: auto;
    }}
    .legend {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px 12px;
      align-items: center;
      color: var(--muted);
      font-size: 12px;
      margin: 10px 0 0;
    }}
    .legend-swatch {{
      display: inline-block;
      width: 18px;
      height: 12px;
      border-radius: 3px;
      background: rgba(var(--heat), 0.58);
      vertical-align: -1px;
      margin-right: 5px;
    }}
    .sample {{
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
      margin: 0 0 10px;
      overflow: hidden;
    }}
    summary {{
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      gap: 8px;
      padding: 10px 12px;
      cursor: pointer;
      background: #f7f9fb;
    }}
    .sample-index {{
      color: var(--muted);
      font-variant-numeric: tabular-nums;
      width: 44px;
    }}
    .sample-id {{
      flex: 1 1 360px;
      font-weight: 700;
      overflow-wrap: anywhere;
    }}
    .sample-body {{
      padding: 12px;
    }}
    .output {{
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
      padding: 10px;
      margin: 0 0 10px;
      font-size: 13px;
    }}
    .token-wall {{
      border: 1px solid var(--line);
      padding: 12px;
      margin-top: 10px;
      background: #fff;
      border-radius: 8px;
      overflow-wrap: anywhere;
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace;
      font-size: 13px;
      line-height: 2.05;
    }}
    .token {{
      display: inline;
      padding: 2px 3px;
      margin: 0 1px 3px 0;
      color: var(--text);
      border-bottom: 1px solid rgba(29, 39, 51, 0.18);
      background: rgba(var(--heat), var(--attn-alpha));
      box-shadow: inset 0 0 0 999px rgba(var(--heat), calc(var(--attn-alpha) * 0.10));
      white-space: pre-wrap;
    }}
    .token.is-hidden {{ display: none; }}
    .region-auth {{ color: var(--auth); }}
    .region-data {{ color: var(--data); }}
    .region-fact {{ color: var(--fact); }}
    .region-attack {{ color: var(--attack); }}
    .region-special {{ color: var(--special); }}
    .token.region-auth,
    .token.region-data,
    .token.region-fact,
    .token.region-attack,
    .token.region-special {{
      color: var(--text);
    }}
    .pill, .badge {{
      display: inline-block;
      border: 1px solid currentColor;
      border-radius: 4px;
      padding: 1px 5px;
      font-size: 11px;
      font-weight: 700;
      line-height: 1.5;
      background: #fff;
    }}
    .badge {{ white-space: nowrap; }}
    .region-score {{
      display: inline-flex;
      gap: 5px;
      align-items: center;
    }}
    .status-good {{ color: var(--good); }}
    .status-bad {{ color: var(--bad); }}
    .status-warn {{ color: var(--warn); }}
    .status-neutral {{ color: var(--muted); }}
    @media (max-width: 980px) {{
      header, main {{ padding-left: 14px; padding-right: 14px; }}
      .summary-grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
      .filter-row {{ grid-template-columns: 1fr; }}
      #visible-count {{ margin-left: 0; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>Attention Tokens - All Samples</h1>
    <div class="meta">
      <span>source <code>{html.escape(source_path)}</code></span>
      <span>samples <code>{len(records)}</code></span>
      <span>render mode <code>all</code></span>
    </div>
    {render_all_summary(records)}
    {render_all_filters(records)}
    <div class="legend"><span><span class="legend-swatch"></span>one shared fill color; stronger fill = higher attention score</span></div>
  </header>
  <main>
    {sample_html}
  </main>
  <script>
    const regionChecks = Array.from(document.querySelectorAll(".region-filter"));
    const minScore = document.getElementById("min-score");
    const scoreSlider = document.getElementById("score-slider");
    const visibleCount = document.getElementById("visible-count");
    const tokens = Array.from(document.querySelectorAll(".token"));

    function selectedRegions() {{
      return new Set(regionChecks.filter((check) => check.checked).map((check) => check.value));
    }}

    function applyFilters() {{
      const regions = selectedRegions();
      const threshold = Number.parseFloat(minScore.value || "0");
      let visible = 0;
      for (const token of tokens) {{
        const region = token.dataset.region;
        const score = Number.parseFloat(token.dataset.score || "0");
        const show = regions.has(region) && score >= threshold;
        token.classList.toggle("is-hidden", !show);
        if (show) visible += 1;
      }}
      visibleCount.textContent = `${{visible}} / ${{tokens.length}} tokens visible`;
    }}

    for (const check of regionChecks) {{
      check.addEventListener("change", applyFilters);
    }}
    minScore.addEventListener("input", () => {{
      scoreSlider.value = minScore.value || "0";
      applyFilters();
    }});
    scoreSlider.addEventListener("input", () => {{
      minScore.value = scoreSlider.value;
      applyFilters();
    }});
    document.getElementById("select-main-regions").addEventListener("click", () => {{
      for (const check of regionChecks) check.checked = true;
      applyFilters();
    }});
    document.getElementById("clear-regions").addEventListener("click", () => {{
      for (const check of regionChecks) check.checked = false;
      applyFilters();
    }});
    document.getElementById("reset-filters").addEventListener("click", () => {{
      for (const check of regionChecks) check.checked = true;
      minScore.value = "0";
      scoreSlider.value = "0";
      applyFilters();
    }});
    applyFilters();
  </script>
</body>
</html>
"""


def render_html(record: dict, source_path: str) -> str:
    tokens = record.get("tokens", [])
    source = record.get("source", {})
    title = f"{record.get('dataset', '')} / {record.get('id', '')}"
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Attention Tokens - {html.escape(str(record.get("id", "")))}</title>
  <style>
    :root {{
      color-scheme: light;
      --text: #1d2733;
      --muted: #667085;
      --line: #d9e0e8;
      --panel: #f7f9fb;
      --heat: 217, 72, 34;
      --auth: #2f6fbd;
      --data: #607080;
      --fact: #21845b;
      --attack: #b42318;
      --special: #7a5c00;
      --good: #177245;
      --bad: #b42318;
      --warn: #9a6700;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: var(--text);
      background: #ffffff;
      line-height: 1.45;
    }}
    .layout {{
      display: grid;
      grid-template-columns: minmax(0, 1fr) 360px;
      min-height: 100vh;
    }}
    main {{
      padding: 24px;
      min-width: 0;
    }}
    aside {{
      border-left: 1px solid var(--line);
      background: var(--panel);
      padding: 20px;
      overflow: auto;
      max-height: 100vh;
      position: sticky;
      top: 0;
    }}
    h1 {{
      font-size: 22px;
      line-height: 1.2;
      margin: 0 0 8px;
      overflow-wrap: anywhere;
    }}
    h2 {{
      font-size: 14px;
      margin: 22px 0 10px;
      text-transform: uppercase;
      color: var(--muted);
      letter-spacing: 0;
    }}
    .meta {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin: 12px 0 18px;
      color: var(--muted);
      font-size: 13px;
    }}
    .meta code {{
      background: #eef2f6;
      border: 1px solid var(--line);
      border-radius: 4px;
      padding: 2px 5px;
      color: var(--text);
    }}
    .status-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
      margin: 14px 0;
    }}
    .status-card {{
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
      background: #fff;
      min-width: 0;
    }}
    .status-card strong {{
      display: block;
      font-size: 20px;
      line-height: 1.2;
      margin: 4px 0;
    }}
    .status-card small, .status-label {{
      color: var(--muted);
      font-size: 12px;
    }}
    .status-good {{ color: var(--good); }}
    .status-bad {{ color: var(--bad); }}
    .status-warn {{ color: var(--warn); }}
    .status-neutral {{ color: var(--muted); }}
    .output {{
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
      padding: 12px;
      margin: 0 0 14px;
      font-size: 13px;
    }}
    .filters {{
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
      padding: 12px;
      margin: 14px 0;
    }}
    .filters h2 {{
      margin: 0 0 8px;
    }}
    .filter-row {{
      display: grid;
      grid-template-columns: minmax(0, 1fr) 260px;
      gap: 16px;
      align-items: end;
    }}
    .checks {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }}
    .check {{
      display: inline-flex;
      align-items: center;
      gap: 5px;
      cursor: pointer;
    }}
    .score-filter {{
      display: grid;
      gap: 6px;
      font-size: 12px;
      color: var(--muted);
    }}
    input[type="number"], input[type="range"] {{
      width: 100%;
    }}
    input[type="number"] {{
      border: 1px solid var(--line);
      border-radius: 5px;
      padding: 6px 7px;
      font: inherit;
      font-variant-numeric: tabular-nums;
    }}
    .filter-actions {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      align-items: center;
      margin-top: 10px;
    }}
    button {{
      border: 1px solid var(--line);
      border-radius: 5px;
      background: #f7f9fb;
      color: var(--text);
      padding: 6px 9px;
      font: inherit;
      font-size: 12px;
      cursor: pointer;
    }}
    #visible-count {{
      color: var(--muted);
      font-size: 12px;
      margin-left: auto;
    }}
    .token-wall {{
      border: 1px solid var(--line);
      padding: 16px;
      background: #fff;
      border-radius: 8px;
      overflow-wrap: anywhere;
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace;
      font-size: 13px;
      line-height: 2.05;
    }}
    .token {{
      display: inline;
      padding: 2px 3px;
      margin: 0 1px 3px 0;
      color: var(--text);
      border-bottom: 1px solid rgba(29, 39, 51, 0.18);
      background: rgba(var(--heat), var(--attn-alpha));
      box-shadow: inset 0 0 0 999px rgba(var(--heat), calc(var(--attn-alpha) * 0.10));
      white-space: pre-wrap;
    }}
    .token.is-hidden {{ display: none; }}
    .legend {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px 12px;
      align-items: center;
      color: var(--muted);
      font-size: 12px;
      margin: 10px 0 14px;
    }}
    .legend-item {{
      display: inline-flex;
      align-items: center;
      gap: 5px;
    }}
    .legend-swatch {{
      width: 18px;
      height: 12px;
      border-radius: 3px;
      background: rgba(var(--heat), 0.58);
    }}
    .legend-note {{
      margin-left: auto;
    }}
    .region-auth {{ color: var(--auth); }}
    .region-data {{ color: var(--data); }}
    .region-fact {{ color: var(--fact); }}
    .region-attack {{ color: var(--attack); }}
    .region-special {{ color: var(--special); }}
    .token.region-auth,
    .token.region-data,
    .token.region-fact,
    .token.region-attack,
    .token.region-special {{
      color: var(--text);
    }}
    .pill {{
      display: inline-block;
      border: 1px solid currentColor;
      border-radius: 4px;
      padding: 1px 5px;
      font-size: 11px;
      font-weight: 700;
      line-height: 1.5;
      background: #fff;
    }}
    .metric {{
      border: 1px solid var(--line);
      background: #fff;
      border-radius: 8px;
      padding: 10px;
      margin-bottom: 8px;
    }}
    .metric-head {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
      font-size: 13px;
    }}
    .metric-foot {{
      color: var(--muted);
      font-size: 12px;
      margin-top: 4px;
    }}
    .bar {{
      height: 7px;
      background: #e7edf3;
      border-radius: 999px;
      overflow: hidden;
      margin-top: 8px;
    }}
    .bar span {{
      display: block;
      height: 100%;
      background: #d94c32;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 12px;
      background: #fff;
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
    }}
    th, td {{
      border-bottom: 1px solid var(--line);
      padding: 7px 8px;
      text-align: left;
      vertical-align: top;
    }}
    th {{
      color: var(--muted);
      font-weight: 700;
      background: #f2f5f8;
    }}
    tr:last-child td {{ border-bottom: 0; }}
    code {{
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace;
      overflow-wrap: anywhere;
    }}
    .num {{
      text-align: right;
      font-variant-numeric: tabular-nums;
    }}
    .note {{
      color: var(--muted);
      font-size: 13px;
      margin-top: 12px;
    }}
    @media (max-width: 920px) {{
      .layout {{ grid-template-columns: 1fr; }}
      aside {{
        border-left: 0;
        border-top: 1px solid var(--line);
        max-height: none;
        position: static;
      }}
      main {{ padding: 16px; }}
    }}
  </style>
</head>
<body>
  <div class="layout">
    <main>
      <h1>{html.escape(title)}</h1>
      <div class="meta">
        <span>source <code>{html.escape(source_path)}</code></span>
        <span>generated token <code>{html.escape(str(record.get("generated_token", "")))}</code></span>
        <span>display tokens <code>{record.get("num_display_tokens", len(tokens))}</code></span>
        <span>filtered tokens <code>{record.get("num_filtered_tokens", 0)}</code></span>
        <span>aggregation <code>{html.escape(str(source.get("aggregation", "")))}</code></span>
      </div>
      {render_status_cards(record)}
      {render_filters(tokens)}
      {render_legend()}
      <div class="token-wall">
        {render_tokens(tokens)}
      </div>
      <p class="note">Fill strength uses one shared scale for all regions and marks attention score. Border color only marks the token region.</p>
    </main>
    <aside>
      <h2>Region Scores</h2>
      {render_region_scores(record.get("region_scores", {}))}

      <h2>Token Ranges</h2>
      <table>
        <thead><tr><th>Region</th><th class="num">Start</th><th class="num">End</th><th class="num">Len</th></tr></thead>
        <tbody>{render_ranges(record.get("token_ranges", {}))}</tbody>
      </table>

      <h2>Top Tokens</h2>
      <table>
        <thead><tr><th class="num">Idx</th><th>Token</th><th>Region</th><th class="num">Score</th></tr></thead>
        <tbody>{render_top_tokens(record.get("top_tokens", []))}</tbody>
      </table>

      <h2>Source</h2>
      <table>
        <tbody>
          <tr><td>Model</td><td><code>{html.escape(str(source.get("model_name", "")))}</code></td></tr>
          <tr><td>Step</td><td><code>{html.escape(str(source.get("attn_step", "")))}</code></td></tr>
          <tr><td>Heads</td><td><code>{len(source.get("important_heads", []))}</code></td></tr>
        </tbody>
      </table>
    </aside>
  </div>
  <script>
    const regionChecks = Array.from(document.querySelectorAll(".region-filter"));
    const minScore = document.getElementById("min-score");
    const scoreSlider = document.getElementById("score-slider");
    const visibleCount = document.getElementById("visible-count");
    const tokens = Array.from(document.querySelectorAll(".token"));

    function selectedRegions() {{
      return new Set(regionChecks.filter((check) => check.checked).map((check) => check.value));
    }}

    function applyFilters() {{
      const regions = selectedRegions();
      const threshold = Number.parseFloat(minScore.value || "0");
      let visible = 0;
      for (const token of tokens) {{
        const region = token.dataset.region;
        const score = Number.parseFloat(token.dataset.score || "0");
        const show = regions.has(region) && score >= threshold;
        token.classList.toggle("is-hidden", !show);
        if (show) visible += 1;
      }}
      visibleCount.textContent = `${{visible}} / ${{tokens.length}} tokens visible`;
    }}

    for (const check of regionChecks) {{
      check.addEventListener("change", applyFilters);
    }}
    minScore.addEventListener("input", () => {{
      scoreSlider.value = minScore.value || "0";
      applyFilters();
    }});
    scoreSlider.addEventListener("input", () => {{
      minScore.value = scoreSlider.value;
      applyFilters();
    }});
    document.getElementById("select-main-regions").addEventListener("click", () => {{
      for (const check of regionChecks) check.checked = true;
      applyFilters();
    }});
    document.getElementById("clear-regions").addEventListener("click", () => {{
      for (const check of regionChecks) check.checked = false;
      applyFilters();
    }});
    document.getElementById("reset-filters").addEventListener("click", () => {{
      for (const check of regionChecks) check.checked = true;
      minScore.value = "0";
      scoreSlider.value = "0";
      applyFilters();
    }});
    applyFilters();
  </script>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Render token-level attention JSONL as an HTML heatmap.")
    parser.add_argument("--input", required=True, help="Attention JSONL produced by run_qwen3_shift_experiment.py.")
    parser.add_argument("--output", required=True, help="HTML output path.")
    parser.add_argument("--results", help="Optional main result JSONL used to annotate AttentionShift and attack status.")
    parser.add_argument("--id", help="Record id to render. If omitted, --index is used.")
    parser.add_argument("--index", type=int, default=0, help="Zero-based record index to render.")
    parser.add_argument("--all", action="store_true", help="Render every attention record into one combined HTML file.")
    parser.add_argument(
        "--repair_regions_from_results",
        action="store_true",
        help="Use sample text from --results and tokenizer offsets to repair FACT/ATTACK token regions.",
    )
    parser.add_argument(
        "--tokenizer_model_id",
        default="/root/Qwen3-8B",
        help="Tokenizer path/model id used with --repair_regions_from_results.",
    )
    parser.add_argument(
        "--write_repaired_attention",
        help="Optional JSONL path for repaired attention records.",
    )
    args = parser.parse_args()

    rows = read_jsonl(args.input)
    result_rows = read_jsonl(args.results) if args.results else None
    tokenizer = None
    if args.repair_regions_from_results:
        if not result_rows:
            raise ValueError("--repair_regions_from_results requires --results.")
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_model_id)
    if args.all:
        records = [drop_special_attention(row) for row in rows]
        records = merge_all_result_metadata(records, result_rows, tokenizer)
        if args.write_repaired_attention:
            write_jsonl(args.write_repaired_attention, records)
        html_text = render_all_html(records, args.input)
    else:
        record = select_record(rows, args.id, args.index)
        record = drop_special_attention(record)
        record = merge_result_metadata(record, result_rows, tokenizer)
        if args.write_repaired_attention:
            write_jsonl(args.write_repaired_attention, [record])
        html_text = render_html(record, args.input)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(html_text)
    print(args.output)


if __name__ == "__main__":
    main()
