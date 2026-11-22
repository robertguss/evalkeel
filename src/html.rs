//! Zero-CDN tamper-evident offline HTML Model Card generator with embedded SVG calibration curves.

use crate::metrics::{BinaryEvalReport, RegressEvalReport};

fn esc(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

const BASE_CSS: &str = r#"
  :root { --bg:#0b1120; --card:#111a2e; --border:#1f2d47; --fg:#e2e8f0; --muted:#94a3b8; }
  * { box-sizing:border-box; }
  body { margin:0; padding:28px; font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif; background:var(--bg); color:var(--fg); line-height:1.5; }
  .wrap { max-width:1100px; margin:0 auto; }
  h1 { margin:0 0 6px; font-size:22px; }
  h2 { margin:0 0 12px; font-size:15px; color:#f8fafc; }
  .sub { color:var(--muted); font-size:13px; margin:0 0 18px; }
  .grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(160px,1fr)); gap:12px; margin-bottom:16px; }
  .two { display:grid; grid-template-columns:repeat(auto-fit,minmax(320px,1fr)); gap:16px; margin-bottom:16px; }
  .kpi { background:var(--card); border:1px solid var(--border); border-radius:10px; padding:14px 16px; }
  .kpi .lbl { font-size:11px; text-transform:uppercase; letter-spacing:.06em; color:var(--muted); }
  .kpi .val { font-size:22px; font-weight:700; margin-top:4px; }
  .card { background:var(--card); border:1px solid var(--border); border-radius:12px; padding:18px 20px; margin-bottom:16px; }
  .pill { display:inline-block; padding:3px 9px; border-radius:999px; font-size:11px; font-weight:700; }
  .pill.ok { background:rgba(16,185,129,.16); color:#34d399; border:1px solid rgba(16,185,129,.35); }
  .pill.bad { background:rgba(239,68,68,.18); color:#f87171; border:1px solid rgba(239,68,68,.4); }
  table { width:100%; border-collapse:collapse; font-size:13px; }
  th, td { padding:8px 10px; border-bottom:1px solid var(--border); text-align:left; }
  th { color:var(--muted); font-size:11px; text-transform:uppercase; letter-spacing:.05em; }
  .num { text-align:right; font-family:ui-monospace,monospace; }
  .foot { margin-top:14px; font-size:11px; color:var(--muted); font-family:ui-monospace,monospace; }
"#;

pub fn render_binary_html(r: &BinaryEvalReport) -> String {
    let gate_pill = if r.gates_passed {
        r#"<span class="pill ok">GATES PASS</span>"#
    } else {
        r#"<span class="pill bad">GATE FAILED</span>"#
    };

    // SVG Reliability / Calibration Diagram (340 x 240)
    let mut svg_pts = Vec::new();
    let mut svg_circles = String::new();
    let pad_l = 44.0_f64;
    let pad_b = 200.0_f64;
    let plot_w = 270.0_f64;
    let plot_h = 175.0_f64;
    for b in &r.calibration_bins {
        if b.count == 0 {
            continue;
        }
        let x = pad_l + b.mean_predicted * plot_w;
        let y = pad_b - b.observed_rate * plot_h;
        svg_pts.push(format!("{x:.1},{y:.1}"));
        let rad = (3.5 + (b.count as f64).ln()).clamp(3.5, 9.0);
        svg_circles.push_str(&format!(
            r##"<circle cx="{x:.1}" cy="{y:.1}" r="{rad:.1}" fill="#38bdf8" stroke="#0b1120" stroke-width="1.5"><title>{}: pred={:.1}%, obs={:.1}% (n={})</title></circle>"##,
            b.range_label,
            b.mean_predicted * 100.0,
            b.observed_rate * 100.0,
            b.count
        ));
    }
    let polyline = if svg_pts.len() >= 2 {
        format!(
            r##"<polyline fill="none" stroke="#38bdf8" stroke-width="2.5" points="{}"/>"##,
            svg_pts.join(" ")
        )
    } else {
        String::new()
    };

    let topk_rows: String = r
        .top_k
        .iter()
        .map(|tk| {
            format!(
                r#"<tr><td>Top {:.0}%</td><td class="num">{}</td><td class="num">{}</td><td class="num">{:.1}%</td><td class="num">{:.1}%</td><td class="num">{:.2}x</td></tr>"#,
                tk.top_k_pct,
                tk.n_flagged,
                tk.positives_captured,
                tk.precision * 100.0,
                tk.recall * 100.0,
                tk.lift
            )
        })
        .collect();

    let comp_html = if let Some(c) = &r.comparison {
        format!(
            r#"<div class="card"><h2>Head-to-Head vs Baseline ({})</h2>
            <p class="sub" style="margin:0">Δ PR-AUC: <strong>{:+.4}</strong> ({:.4} vs {:.4}) · Δ ROC-AUC: <strong>{:+.4}</strong> · Δ Brier: <strong>{:+.4}</strong></p></div>"#,
            esc(&c.baseline_col),
            c.delta_pr_auc,
            r.pr_auc,
            c.baseline_pr_auc,
            c.delta_roc_auc,
            c.delta_brier
        )
    } else {
        String::new()
    };

    let fair_html = if let Some(f) = &r.fairness {
        let rows: String = f
            .groups
            .iter()
            .map(|g| {
                format!(
                    r#"<tr><td><strong>{}</strong></td><td class="num">{}</td><td class="num">{:.1}%</td><td class="num">{}</td><td class="num">{}</td><td class="num">{:.4}</td><td class="num">{:.1}%</td></tr>"#,
                    esc(&g.group),
                    g.n,
                    g.base_rate * 100.0,
                    g.roc_auc.map(|v| format!("{v:.4}")).unwrap_or_else(|| "—".into()),
                    g.pr_auc.map(|v| format!("{v:.4}")).unwrap_or_else(|| "—".into()),
                    g.ece,
                    g.tpr * 100.0
                )
            })
            .collect();
        format!(
            r#"<div class="card"><h2>Subgroup Fairness &amp; Calibration Parity ({})</h2>
            <p class="sub">Max TPR gap: <strong>{:.1}pp</strong> · Max ECE gap: <strong>{:.4}</strong> · Max AUC gap: <strong>{:.4}</strong></p>
            <table><thead><tr><th>Group</th><th class="num">N</th><th class="num">Base Rate</th><th class="num">ROC-AUC</th><th class="num">PR-AUC</th><th class="num">ECE</th><th class="num">TPR @{:.2}</th></tr></thead><tbody>{rows}</tbody></table></div>"#,
            esc(&f.group_col),
            f.max_tpr_gap * 100.0,
            f.max_ece_gap,
            f.max_auc_gap,
            f.threshold
        )
    } else {
        String::new()
    };

    format!(
        r##"<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{}</title><style>{BASE_CSS}</style></head><body><div class="wrap">
  <h1>{} {gate_pill}</h1>
  <p class="sub">Source: <strong>{}</strong> · Evaluated <strong>{}</strong> rows ({} positives, base rate {:.1}%) · Generated {}</p>
  <div class="grid">
    <div class="kpi"><div class="lbl">PR-AUC (Avg Prec)</div><div class="val">{:.4}</div></div>
    <div class="kpi"><div class="lbl">ROC-AUC</div><div class="val">{:.4}</div></div>
    <div class="kpi"><div class="lbl">Brier Score</div><div class="val">{:.4}</div></div>
    <div class="kpi"><div class="lbl">Calibration ECE</div><div class="val">{:.4}</div></div>
    <div class="kpi"><div class="lbl">Log-Loss</div><div class="val">{:.4}</div></div>
  </div>
  {comp_html}
  <div class="two">
    <div class="card">
      <h2>Reliability / Calibration Diagram</h2>
      <svg viewBox="0 0 340 230" width="100%" style="max-width:380px">
        <line x1="44" y1="200" x2="314" y2="200" stroke="#334155"/>
        <line x1="44" y1="25" x2="44" y2="200" stroke="#334155"/>
        <line x1="44" y1="200" x2="314" y2="25" stroke="#64748b" stroke-dasharray="4,4"/>
        {polyline}
        {svg_circles}
        <text x="179" y="222" fill="#94a3b8" font-size="11" text-anchor="middle">Mean Predicted Probability</text>
        <text x="14" y="115" fill="#94a3b8" font-size="11" text-anchor="middle" transform="rotate(-90 14 115)">Observed Rate</text>
      </svg>
    </div>
    <div class="card">
      <h2>Operational Top-K% Capture &amp; Lift</h2>
      <table><thead><tr><th>Bucket</th><th class="num">Flagged</th><th class="num">Hits</th><th class="num">Prec</th><th class="num">Recall</th><th class="num">Lift</th></tr></thead>
      <tbody>{topk_rows}</tbody></table>
    </div>
  </div>
  {fair_html}
  <div class="foot">evalkeel v{} · tamper-evident receipt sha256: {}</div>
</div></body></html>"##,
        esc(&r.title),
        esc(&r.title),
        esc(&r.source),
        r.n_rows,
        r.positives,
        r.base_rate * 100.0,
        esc(&r.generated_at),
        r.pr_auc,
        r.roc_auc,
        r.brier_score,
        r.ece,
        r.log_loss,
        esc(&r.version),
        esc(&r.sha256)
    )
}

pub fn render_regress_html(r: &RegressEvalReport) -> String {
    let gate_pill = if r.gates_passed {
        r#"<span class="pill ok">GATES PASS</span>"#
    } else {
        r#"<span class="pill bad">GATE FAILED</span>"#
    };
    let conf_html = if let Some(c) = &r.conformal {
        format!(
            r#"<div class="card"><h2>Conformal Prediction Interval ({} – {})</h2>
            <p class="sub" style="margin:0">Empirical Coverage: <strong>{:.1}%</strong> · Below P10: <strong>{:.1}%</strong> · Above P90: <strong>{:.1}%</strong> · Mean Width: <strong>{:.2}</strong></p></div>"#,
            esc(&c.p10_col),
            esc(&c.p90_col),
            c.empirical_coverage * 100.0,
            c.below_p10_rate * 100.0,
            c.above_p90_rate * 100.0,
            c.mean_interval_width
        )
    } else {
        String::new()
    };
    let cap_html = if let Some(cp) = &r.capacity {
        format!(
            r#"<div class="card"><h2>Capacity &amp; Section Staffing Accuracy (Cap = {:.0}/section)</h2>
            <p class="sub" style="margin:0">Exact Section Match: <strong>{:.1}%</strong> · Within ±1 Section: <strong>{:.1}%</strong> · Under-staffed: <strong>{:.1}%</strong> · Total Sections: <strong>{}</strong> predicted vs <strong>{}</strong> actual</p></div>"#,
            cp.cap_per_section,
            cp.exact_section_accuracy * 100.0,
            cp.within_one_section_accuracy * 100.0,
            cp.under_staffed_rate * 100.0,
            cp.total_pred_sections,
            cp.total_actual_sections
        )
    } else {
        String::new()
    };

    format!(
        r#"<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{}</title><style>{BASE_CSS}</style></head><body><div class="wrap">
  <h1>{} {gate_pill}</h1>
  <p class="sub">Source: <strong>{}</strong> · Evaluated <strong>{}</strong> rows · Generated {}</p>
  <div class="grid">
    <div class="kpi"><div class="lbl">WMAPE</div><div class="val">{:.1}%</div></div>
    <div class="kpi"><div class="lbl">MAE</div><div class="val">{:.2}</div></div>
    <div class="kpi"><div class="lbl">RMSE</div><div class="val">{:.2}</div></div>
    <div class="kpi"><div class="lbl">R²</div><div class="val">{:.4}</div></div>
    <div class="kpi"><div class="lbl">Bias (Pred-Act)</div><div class="val">{:+.2}</div></div>
  </div>
  {conf_html}
  {cap_html}
  <div class="foot">evalkeel v{} · tamper-evident receipt sha256: {}</div>
</div></body></html>"#,
        esc(&r.title),
        esc(&r.title),
        esc(&r.source),
        r.n_rows,
        esc(&r.generated_at),
        r.wmape * 100.0,
        r.mae,
        r.rmse,
        r.r2,
        r.bias,
        esc(&r.version),
        esc(&r.sha256)
    )
}
