import argparse
import html
import json
import math
import os


REGION_LABELS = {
    "auth": "AUTH",
    "data": "DATA",
    "data_fact": "FACT",
    "data_attack": "ATTACK",
}

REGION_CLASSES = {
    "auth": "region-auth",
    "data": "region-data",
    "data_fact": "region-fact",
    "data_attack": "region-attack",
}


def read_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def select_record(rows: list[dict], record_id: str | None, index: int) -> dict:
    if record_id is not None:
        for row in rows:
            if str(row.get("id")) == record_id:
                return row
        raise ValueError(f"No record found with id={record_id!r}.")
    if index < 0 or index >= len(rows):
        raise IndexError(f"--index {index} is out of range for {len(rows)} records.")
    return rows[index]


def bool_label(value) -> str:
    if value is True:
        return "YES"
    if value is False:
        return "NO"
    return "UNKNOWN"


def status_class(value) -> str:
    text = str(value).lower()
    if text in ("true", "success"):
        return "status-bad"
    if text in ("false", "failed"):
        return "status-good"
    if text in ("ambiguous", "skipped", "unjudged", "none"):
        return "status-warn"
    return "status-neutral"


def value_scale(values: list[float]) -> tuple[float, float]:
    if not values:
        return -1.0, 1.0
    lo = min(values)
    hi = max(values)
    if math.isclose(lo, hi):
        spread = max(abs(lo), 1.0)
        return -spread, spread
    bound = max(abs(lo), abs(hi))
    return -bound, bound


def phi_alpha(value: float, bound: float) -> float:
    if bound <= 1e-12:
        return 0.18
    scaled = min(abs(value) / bound, 1.0)
    return 0.14 + scaled * 0.78


def fmt_float(value) -> str:
    if value is None:
        return "null"
    return f"{float(value):.6f}"


def render_prompt(record: dict) -> str:
    sample = record.get("sample") or {}
    sections = [
        ("auth", f"<system>\n{sample.get('system', '')}\n</system>\n\n<user>\n{sample.get('user', '')}\n</user>"),
        ("data_fact", sample.get("data_fact", "")),
        ("data_attack", sample.get("data_attack", "")),
    ]
    blocks = []
    for region, text in sections:
        label = REGION_LABELS[region]
        cls = REGION_CLASSES[region]
        safe_text = html.escape(str(text)) if text else "<span class=\"empty\">(empty)</span>"
        blocks.append(
            f"""
            <section class="prompt-block {cls}">
              <div class="prompt-head">
                <span class="pill {cls}">{label}</span>
              </div>
              <pre>{safe_text}</pre>
            </section>
            """
        )
    return "\n".join(blocks)


def shapley_payload(record: dict, granularity: str) -> dict | None:
    return record.get(f"shapley_{granularity}")


def player_score(payload: dict | None, player: str) -> float | None:
    if not payload:
        return None
    phi = payload.get("phi") or {}
    if player not in phi:
        return None
    return float(phi[player])


def format_player_scores(payload: dict | None) -> str:
    if not payload:
        return "N/A"
    phi = payload.get("phi") or {}
    ordered = payload.get("players") or list(phi.keys())
    parts = []
    for player in ordered:
        if player not in phi:
            continue
        parts.append(f"{REGION_LABELS.get(player, player.upper())} {float(phi[player]):.4f}")
    return " | ".join(parts) if parts else "N/A"


def should_expand_payload(payload: dict | None) -> bool:
    if not payload:
        return False
    players = list(payload.get("players") or [])
    if players == ["auth", "data"] or set(players) == {"auth", "data"}:
        auth = player_score(payload, "auth")
        data = player_score(payload, "data")
        return auth is not None and data is not None and data > auth
    if set(players) == {"auth", "data_fact", "data_attack"}:
        auth = player_score(payload, "auth")
        fact = player_score(payload, "data_fact")
        attack = player_score(payload, "data_attack")
        if auth is None or fact is None or attack is None:
            return False
        return (fact + attack) > auth and attack > auth
    return False


def should_expand_row(row: dict, granularities: list[str]) -> bool:
    for granularity in granularities:
        if should_expand_payload(shapley_payload(row, granularity)):
            return True
    return False


def render_phi_cards(payload: dict) -> str:
    phi = payload.get("phi") or {}
    bound = max((abs(float(v)) for v in phi.values()), default=1.0)
    cards = []
    for player, value in sorted(phi.items(), key=lambda item: abs(float(item[1])), reverse=True):
        value = float(value)
        cls = "phi-pos" if value >= 0 else "phi-neg"
        region_cls = REGION_CLASSES.get(player, "region-data")
        alpha = phi_alpha(value, bound)
        direction = "positive contribution" if value >= 0 else "negative contribution"
        cards.append(
            f"""
            <div class="phi-card {cls}" style="--phi-alpha:{alpha:.4f}">
              <div class="phi-head">
                <span class="pill {region_cls}">{html.escape(REGION_LABELS.get(player, player.upper()))}</span>
                <span class="phi-direction">{direction}</span>
              </div>
              <strong>{value:.6f}</strong>
            </div>
            """
        )
    return "\n".join(cards)


def render_phi_bars(payload: dict) -> str:
    phi = payload.get("phi") or {}
    values = [float(v) for v in phi.values()]
    _, hi = value_scale(values)
    rows = []
    for player, value in sorted(phi.items(), key=lambda item: abs(float(item[1])), reverse=True):
        value = float(value)
        width = 0.0 if hi <= 1e-12 else min(abs(value) / hi * 100.0, 100.0)
        direction_cls = "fill-pos" if value >= 0 else "fill-neg"
        region_cls = REGION_CLASSES.get(player, "region-data")
        rows.append(
            f"""
            <div class="bar-row">
              <div class="bar-label">
                <span class="pill {region_cls}">{html.escape(REGION_LABELS.get(player, player.upper()))}</span>
                <span class="bar-value">{value:.6f}</span>
              </div>
              <div class="bar-track">
                <span class="bar-fill {direction_cls}" style="width:{width:.2f}%"></span>
              </div>
            </div>
            """
        )
    return "\n".join(rows)


def coalition_sort_key(item: tuple[str, float]) -> tuple[int, str]:
    name, _ = item
    size = 0 if name == "empty" else len(name.split("+"))
    return size, name


def render_coalition_table(payload: dict) -> str:
    values = payload.get("values") or {}
    rows = []
    for coalition, score in sorted(values.items(), key=coalition_sort_key):
        players = [] if coalition == "empty" else coalition.split("+")
        player_html = "".join(
            f'<span class="pill {REGION_CLASSES.get(player, "region-data")}">{html.escape(REGION_LABELS.get(player, player.upper()))}</span>'
            for player in players
        )
        if not player_html:
            player_html = '<span class="pill region-empty">EMPTY</span>'
        rows.append(
            f"""
            <tr>
              <td>{player_html}</td>
              <td class="num">{len(players)}</td>
              <td class="num">{float(score):.6f}</td>
            </tr>
            """
        )
    return "\n".join(rows)


def render_granularity(record: dict, granularity: str) -> str:
    payload = shapley_payload(record, granularity)
    if not payload:
        return f"""
        <section class="panel">
          <div class="section-head">
            <h2>{html.escape(granularity.title())} Shapley</h2>
          </div>
          <p class="empty-note">No `shapley_{html.escape(granularity)}` data found in this record.</p>
        </section>
        """
    return f"""
    <section class="panel">
      <div class="section-head">
        <h2>{html.escape(granularity.title())} Shapley</h2>
        <span class="meta-chip">{len(payload.get("players", []))} players</span>
        <span class="meta-chip">{len(payload.get("values", {}))} coalitions</span>
      </div>
      <div class="phi-grid">
        {render_phi_cards(payload)}
      </div>
      <div class="panel-split">
        <div>
          <h3>Contribution Magnitude</h3>
          {render_phi_bars(payload)}
        </div>
        <div>
          <h3>Coalition Values</h3>
          <div class="table-wrap">
            <table>
              <thead>
                <tr><th>Coalition</th><th>Size</th><th>v(S)</th></tr>
              </thead>
              <tbody>
                {render_coalition_table(payload)}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </section>
    """


def render_status_cards(record: dict) -> str:
    attention = record.get("attention") or {}
    output = record.get("output")
    output_html = ""
    if output:
        output_html = f'<div class="output"><strong>Output</strong><br>{html.escape(str(output))}</div>'
    return f"""
    <div class="status-grid">
      <div class="status-card">
        <span class="status-label">Attention Shift</span>
        <strong class="{status_class(record.get('attention_shift'))}">{bool_label(record.get('attention_shift'))}</strong>
        <small>focus {fmt_float(attention.get('focus_score'))} / threshold {fmt_float(attention.get('threshold'))}</small>
      </div>
      <div class="status-card">
        <span class="status-label">Attack Status</span>
        <strong class="{status_class(record.get('injection_status'))}">{html.escape(str(record.get('injection_status', 'unknown')).upper())}</strong>
        <small>candidate {bool_label(record.get('candidate_attention_shift_attack_failed'))}</small>
      </div>
    </div>
    {output_html}
    """


def render_single_html(record: dict, source_path: str, granularities: list[str]) -> str:
    granularity_html = "\n".join(render_granularity(record, granularity) for granularity in granularities)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Shapley Visualization - {html.escape(str(record.get("id", "")))}</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #fcfbf7;
      --panel: rgba(255, 255, 255, 0.92);
      --line: #d8d4c8;
      --text: #1f2933;
      --muted: #667085;
      --accent: #c7672b;
      --auth: #1f5ea8;
      --data: #6b7280;
      --fact: #1f8a5b;
      --attack: #b42318;
      --good: #177245;
      --bad: #b42318;
      --warn: #9a6700;
      --shadow: 0 18px 40px rgba(78, 55, 31, 0.08);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Georgia, "Iowan Old Style", "Palatino Linotype", serif;
      color: var(--text);
      background:
        radial-gradient(circle at top left, rgba(199, 103, 43, 0.10), transparent 24%),
        radial-gradient(circle at top right, rgba(31, 138, 91, 0.08), transparent 28%),
        linear-gradient(180deg, #f8f5ed 0%, var(--bg) 100%);
      line-height: 1.5;
    }}
    .shell {{
      max-width: 1380px;
      margin: 0 auto;
      padding: 28px 20px 40px;
    }}
    header {{
      margin-bottom: 18px;
      padding: 26px;
      border: 1px solid var(--line);
      background: var(--panel);
      box-shadow: var(--shadow);
      border-radius: 22px;
      backdrop-filter: blur(10px);
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: clamp(28px, 4vw, 42px);
      line-height: 1.05;
      letter-spacing: -0.03em;
    }}
    h2 {{
      margin: 0;
      font-size: 24px;
      line-height: 1.15;
    }}
    h3 {{
      margin: 0 0 12px;
      font-size: 15px;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.04em;
      font-family: ui-sans-serif, system-ui, sans-serif;
    }}
    .lede {{
      color: var(--muted);
      max-width: 900px;
      font-size: 16px;
      margin: 0 0 16px;
    }}
    .meta {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      color: var(--muted);
      font-family: ui-sans-serif, system-ui, sans-serif;
      font-size: 13px;
    }}
    .meta code {{
      padding: 2px 6px;
      border-radius: 999px;
      background: #f2eee5;
      border: 1px solid var(--line);
      color: var(--text);
    }}
    .layout {{
      display: grid;
      grid-template-columns: minmax(320px, 430px) minmax(0, 1fr);
      gap: 18px;
      align-items: start;
    }}
    .stack {{
      display: grid;
      gap: 18px;
    }}
    .panel {{
      border: 1px solid var(--line);
      background: var(--panel);
      box-shadow: var(--shadow);
      border-radius: 22px;
      padding: 20px;
      backdrop-filter: blur(10px);
    }}
    .section-head {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      align-items: center;
      margin-bottom: 14px;
    }}
    .meta-chip {{
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 4px 9px;
      font: 12px ui-sans-serif, system-ui, sans-serif;
      color: var(--muted);
      background: #faf8f2;
    }}
    .status-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 12px;
      margin-bottom: 14px;
    }}
    .status-card {{
      border: 1px solid var(--line);
      border-radius: 16px;
      background: #fffdfa;
      padding: 14px;
      font-family: ui-sans-serif, system-ui, sans-serif;
    }}
    .status-label {{
      display: block;
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.04em;
      margin-bottom: 5px;
    }}
    .status-card strong {{
      font-size: 22px;
      display: block;
      line-height: 1.1;
    }}
    .status-card small {{
      color: var(--muted);
    }}
    .output {{
      border: 1px solid var(--line);
      border-radius: 16px;
      padding: 14px;
      background: #fffdfa;
      font-size: 14px;
      overflow-wrap: anywhere;
    }}
    .prompt-block {{
      border: 1px solid var(--line);
      border-radius: 18px;
      margin-bottom: 12px;
      overflow: hidden;
      background: #fffdfa;
    }}
    .prompt-head {{
      padding: 10px 12px;
      border-bottom: 1px solid var(--line);
      background: rgba(0, 0, 0, 0.02);
    }}
    .prompt-block pre {{
      margin: 0;
      padding: 14px;
      white-space: pre-wrap;
      word-break: break-word;
      font: 13px/1.65 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    }}
    .empty {{
      color: var(--muted);
      font-style: italic;
    }}
    .phi-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 12px;
      margin-bottom: 16px;
    }}
    .phi-card {{
      border: 1px solid var(--line);
      border-radius: 16px;
      padding: 14px;
      background: #fffdfa;
      position: relative;
      overflow: hidden;
    }}
    .phi-card::after {{
      content: "";
      position: absolute;
      inset: auto 0 0 0;
      height: 8px;
      opacity: var(--phi-alpha);
    }}
    .phi-card.phi-pos::after {{
      background: linear-gradient(90deg, rgba(31, 138, 91, 0.25), rgba(31, 138, 91, 0.95));
    }}
    .phi-card.phi-neg::after {{
      background: linear-gradient(90deg, rgba(180, 35, 24, 0.25), rgba(180, 35, 24, 0.95));
    }}
    .phi-head {{
      display: flex;
      justify-content: space-between;
      gap: 10px;
      align-items: center;
      margin-bottom: 12px;
      font-family: ui-sans-serif, system-ui, sans-serif;
    }}
    .phi-direction {{
      color: var(--muted);
      font-size: 12px;
    }}
    .phi-card strong {{
      display: block;
      font-size: 28px;
      line-height: 1.05;
    }}
    .panel-split {{
      display: grid;
      grid-template-columns: minmax(280px, 0.9fr) minmax(0, 1.1fr);
      gap: 18px;
      align-items: start;
    }}
    .bar-row {{
      margin-bottom: 12px;
      font-family: ui-sans-serif, system-ui, sans-serif;
    }}
    .bar-label {{
      display: flex;
      justify-content: space-between;
      gap: 10px;
      align-items: center;
      margin-bottom: 6px;
      font-size: 13px;
    }}
    .bar-value {{
      font-variant-numeric: tabular-nums;
      color: var(--muted);
    }}
    .bar-track {{
      height: 12px;
      border-radius: 999px;
      background: #eee7db;
      overflow: hidden;
      border: 1px solid #e1d7c6;
    }}
    .bar-fill {{
      display: block;
      height: 100%;
      border-radius: inherit;
    }}
    .fill-pos {{
      background: linear-gradient(90deg, #87d3ae, #1f8a5b);
    }}
    .fill-neg {{
      background: linear-gradient(90deg, #f6b0aa, #b42318);
    }}
    .table-wrap {{
      overflow: auto;
      border: 1px solid var(--line);
      border-radius: 16px;
      background: #fffdfa;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font: 13px/1.45 ui-sans-serif, system-ui, sans-serif;
    }}
    th, td {{
      padding: 10px 12px;
      border-bottom: 1px solid #eee7db;
      vertical-align: top;
    }}
    th {{
      text-align: left;
      background: #f7f1e5;
      color: var(--muted);
      position: sticky;
      top: 0;
    }}
    .num {{
      font-variant-numeric: tabular-nums;
      white-space: nowrap;
    }}
    .pill {{
      display: inline-block;
      border-radius: 999px;
      padding: 4px 9px;
      border: 1px solid currentColor;
      font: 12px ui-sans-serif, system-ui, sans-serif;
      font-weight: 700;
      background: rgba(255, 255, 255, 0.85);
      margin-right: 6px;
      margin-bottom: 6px;
    }}
    .region-auth {{ color: var(--auth); }}
    .region-data {{ color: var(--data); }}
    .region-fact {{ color: var(--fact); }}
    .region-attack {{ color: var(--attack); }}
    .region-empty {{ color: var(--muted); }}
    .status-good {{ color: var(--good); }}
    .status-bad {{ color: var(--bad); }}
    .status-warn {{ color: var(--warn); }}
    .status-neutral {{ color: var(--muted); }}
    .empty-note {{
      color: var(--muted);
      font-family: ui-sans-serif, system-ui, sans-serif;
      margin: 0;
    }}
    @media (max-width: 980px) {{
      .shell {{ padding: 16px 12px 28px; }}
      .layout {{
        grid-template-columns: 1fr;
      }}
      .panel-split {{
        grid-template-columns: 1fr;
      }}
      .status-grid {{
        grid-template-columns: 1fr;
      }}
    }}
  </style>
</head>
<body>
  <div class="shell">
    <header>
      <h1>Shapley Visualization</h1>
      <p class="lede">This view shows how each prompt region contributes to the final output log-probability under embedding-level masking. Positive values support the observed output; negative values suppress it.</p>
      <div class="meta">
        <span>source <code>{html.escape(source_path)}</code></span>
        <span>dataset <code>{html.escape(str(record.get("dataset", "")))}</code></span>
        <span>id <code>{html.escape(str(record.get("id", "")))}</code></span>
      </div>
    </header>
    <div class="layout">
      <div class="stack">
        <section class="panel">
          <div class="section-head"><h2>Result Status</h2></div>
          {render_status_cards(record)}
        </section>
        <section class="panel">
          <div class="section-head"><h2>Prompt Regions</h2></div>
          {render_prompt(record)}
        </section>
      </div>
      <div class="stack">
        {granularity_html}
      </div>
    </div>
  </div>
</body>
</html>
"""


def render_summary_cards(rows: list[dict]) -> str:
    total = len(rows)
    with_coarse = sum(1 for row in rows if row.get("shapley_coarse"))
    with_fine = sum(1 for row in rows if row.get("shapley_fine"))
    attention_shift = sum(1 for row in rows if row.get("attention_shift") is True)
    candidates = sum(1 for row in rows if row.get("candidate_attention_shift_attack_failed") is True)
    success = sum(1 for row in rows if row.get("injection_status") == "success")
    return f"""
    <div class="summary-grid">
      <div class="summary-card"><span>Total</span><strong>{total}</strong></div>
      <div class="summary-card"><span>Coarse Ready</span><strong>{with_coarse}</strong></div>
      <div class="summary-card"><span>Fine Ready</span><strong>{with_fine}</strong></div>
      <div class="summary-card"><span>Attention Shift</span><strong>{attention_shift}</strong></div>
      <div class="summary-card"><span>Shift + Failed</span><strong>{candidates}</strong></div>
      <div class="summary-card"><span>Attack Success</span><strong>{success}</strong></div>
    </div>
    """


def dominant_player(payload: dict | None) -> str:
    if not payload or not payload.get("phi"):
        return "N/A"
    player, value = max(payload["phi"].items(), key=lambda item: abs(float(item[1])))
    return f"{REGION_LABELS.get(player, player.upper())} ({float(value):.4f})"


def render_gallery_row(row: dict, index: int, granularities: list[str]) -> str:
    tags = []
    for granularity in granularities:
        payload = shapley_payload(row, granularity)
        if payload:
            tags.append(
                f'<span class="badge">{html.escape(granularity.title())}: {html.escape(dominant_player(payload))}</span>'
            )
            tags.append(
                f'<span class="badge badge-score">{html.escape(format_player_scores(payload))}</span>'
            )
    open_attr = " open" if should_expand_row(row, granularities) else ""
    output = row.get("output")
    output_html = ""
    if output:
        output_html = (
            f'<div class="gallery-output"><strong>Model Output</strong>'
            f'<pre>{html.escape(str(output))}</pre></div>'
        )
    return f"""
    <details class="sample"{open_attr}>
      <summary>
        <span class="sample-index">#{index}</span>
        <span class="sample-id">{html.escape(str(row.get("dataset", "")))} / {html.escape(str(row.get("id", "")))}</span>
        <span class="badge {status_class(row.get('attention_shift'))}">Shift {bool_label(row.get('attention_shift'))}</span>
        <span class="badge {status_class(row.get('injection_status'))}">Attack {html.escape(str(row.get('injection_status', 'unknown')).upper())}</span>
        {''.join(tags)}
      </summary>
      <div class="sample-body">
        {output_html}
        <div class="prompt-mini">
          <div><strong>System + User</strong><pre>{html.escape(str((row.get("sample") or {}).get("system", "")))}{"\\n\\n"}{html.escape(str((row.get("sample") or {}).get("user", "")))}</pre></div>
          <div><strong>Fact</strong><pre>{html.escape(str((row.get("sample") or {}).get("data_fact", "")))}</pre></div>
          <div><strong>Attack</strong><pre>{html.escape(str((row.get("sample") or {}).get("data_attack", "")))}</pre></div>
        </div>
      </div>
    </details>
    """


def render_gallery_html(rows: list[dict], source_path: str, granularities: list[str]) -> str:
    rendered_rows = "\n".join(render_gallery_row(row, i, granularities) for i, row in enumerate(rows))
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Shapley Visualization - Gallery</title>
  <style>
    :root {{
      --text: #1f2933;
      --muted: #667085;
      --line: #d8d4c8;
      --panel: #fffdfa;
      --bg: #f8f5ed;
      --good: #177245;
      --bad: #b42318;
      --warn: #9a6700;
      --shadow: 0 18px 40px rgba(78, 55, 31, 0.08);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: ui-sans-serif, system-ui, sans-serif;
      color: var(--text);
      background: linear-gradient(180deg, #f7f3eb 0%, var(--bg) 100%);
    }}
    .shell {{
      max-width: 1320px;
      margin: 0 auto;
      padding: 22px 16px 34px;
    }}
    header, .sample {{
      border: 1px solid var(--line);
      border-radius: 18px;
      background: var(--panel);
      box-shadow: var(--shadow);
    }}
    header {{
      padding: 22px;
      margin-bottom: 16px;
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: 34px;
      letter-spacing: -0.03em;
    }}
    .meta {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      color: var(--muted);
      font-size: 13px;
    }}
    code {{
      padding: 2px 6px;
      border-radius: 999px;
      border: 1px solid var(--line);
      background: #f2eee5;
    }}
    .summary-grid {{
      display: grid;
      grid-template-columns: repeat(6, minmax(0, 1fr));
      gap: 10px;
      margin-top: 14px;
    }}
    .summary-card {{
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 12px;
      background: #fff;
    }}
    .summary-card span {{
      color: var(--muted);
      display: block;
      font-size: 12px;
      text-transform: uppercase;
    }}
    .summary-card strong {{
      display: block;
      font-size: 24px;
      line-height: 1.1;
      margin-top: 4px;
    }}
    .sample {{
      margin-bottom: 12px;
      overflow: hidden;
    }}
    summary {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      align-items: center;
      padding: 12px 14px;
      cursor: pointer;
      background: #f7f1e5;
    }}
    .sample-index {{
      width: 44px;
      color: var(--muted);
      font-variant-numeric: tabular-nums;
    }}
    .sample-id {{
      flex: 1 1 320px;
      font-weight: 700;
      overflow-wrap: anywhere;
    }}
    .sample-body {{
      padding: 14px;
    }}
    .gallery-output {{
      border: 1px solid var(--line);
      border-radius: 14px;
      background: #fff;
      padding: 12px;
      margin-bottom: 12px;
    }}
    .gallery-output strong {{
      display: block;
      margin-bottom: 8px;
    }}
    .gallery-output pre {{
      margin: 0;
      white-space: pre-wrap;
      word-break: break-word;
      font: 12px/1.6 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      max-height: 280px;
      overflow: auto;
    }}
    .prompt-mini {{
      display: grid;
      grid-template-columns: 1.2fr 1fr 1fr;
      gap: 12px;
    }}
    .prompt-mini div {{
      border: 1px solid var(--line);
      border-radius: 14px;
      background: #fff;
      padding: 12px;
    }}
    .prompt-mini pre {{
      margin: 8px 0 0;
      white-space: pre-wrap;
      word-break: break-word;
      font: 12px/1.55 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      max-height: 320px;
      overflow: auto;
    }}
    .badge {{
      display: inline-block;
      border: 1px solid currentColor;
      border-radius: 999px;
      padding: 3px 8px;
      font-size: 11px;
      font-weight: 700;
      background: #fff;
      color: var(--muted);
    }}
    .badge-score {{
      max-width: 100%;
      white-space: normal;
      font-variant-numeric: tabular-nums;
    }}
    .status-good {{ color: var(--good); }}
    .status-bad {{ color: var(--bad); }}
    .status-warn {{ color: var(--warn); }}
    .status-neutral {{ color: var(--muted); }}
    @media (max-width: 980px) {{
      .summary-grid {{
        grid-template-columns: repeat(2, minmax(0, 1fr));
      }}
      .prompt-mini {{
        grid-template-columns: 1fr;
      }}
    }}
  </style>
</head>
<body>
  <div class="shell">
    <header>
      <h1>Shapley Gallery</h1>
      <div class="meta">
        <span>source <code>{html.escape(source_path)}</code></span>
        <span>samples <code>{len(rows)}</code></span>
        <span>granularity <code>{html.escape(",".join(granularities))}</code></span>
      </div>
      {render_summary_cards(rows)}
    </header>
    {rendered_rows}
  </div>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render HTML visualization for shapley_coarse / shapley_fine fields in experiment JSONL."
    )
    parser.add_argument("--input", required=True, help="Input JSONL containing Shapley results.")
    parser.add_argument("--output", required=True, help="Output HTML path.")
    parser.add_argument("--id", help="Optional record id to render.")
    parser.add_argument("--index", type=int, default=0, help="Record index when --id is not provided.")
    parser.add_argument(
        "--granularity",
        choices=["both", "coarse", "fine"],
        default="both",
        help="Which Shapley granularity to render.",
    )
    parser.add_argument(
        "--mode",
        choices=["single", "gallery"],
        default="single",
        help="Render one sample or a gallery of all samples.",
    )
    parser.add_argument("--limit", type=int, help="Optional limit for gallery mode.")
    args = parser.parse_args()

    rows = read_jsonl(args.input)
    granularities = ["coarse", "fine"] if args.granularity == "both" else [args.granularity]

    if args.mode == "gallery":
        gallery_rows = rows[: args.limit] if args.limit is not None else rows
        html_text = render_gallery_html(gallery_rows, args.input, granularities)
    else:
        record = select_record(rows, args.id, args.index)
        html_text = render_single_html(record, args.input, granularities)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(html_text)


if __name__ == "__main__":
    main()
