//! Exact ML evaluation metrics matching `scikit-learn`:
//! ROC-AUC (exact tie handling), PR-AUC / Average Precision, Brier, Brier Skill Score,
//! Log-Loss, 10-bin ECE reliability curve, Top-K% Capture/Lift, Subgroup Fairness Parity,
//! Regression MAE/RMSE/MedAE/WMAPE/R2, Conformal P10-P90 coverage, and capacity/section staffing.

use std::collections::BTreeMap;

use chrono::Utc;
use serde::Serialize;
use sha2::{Digest, Sha256};

use crate::io_data::DataTable;

fn r4(v: f64) -> f64 {
    (v * 10_000.0).round() / 10_000.0
}

fn r6(v: f64) -> f64 {
    (v * 1_000_000.0).round() / 1_000_000.0
}

#[derive(Debug, Clone, Serialize)]
pub struct GateResult {
    pub metric: String,
    pub rule: String,
    pub actual: f64,
    pub passed: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct CalibrationBin {
    pub bin_index: usize,
    pub range_label: String,
    pub count: usize,
    pub mean_predicted: f64,
    pub observed_rate: f64,
    pub abs_gap: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct TopKMetric {
    pub top_k_pct: f64,
    pub n_flagged: usize,
    pub positives_captured: usize,
    pub precision: f64,
    pub recall: f64,
    pub lift: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct BinaryCompare {
    pub baseline_col: String,
    pub baseline_roc_auc: f64,
    pub baseline_pr_auc: f64,
    pub baseline_brier: f64,
    pub delta_roc_auc: f64,
    pub delta_pr_auc: f64,
    pub delta_brier: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct SubgroupBinaryMetric {
    pub group: String,
    pub n: usize,
    pub positives: usize,
    pub base_rate: f64,
    pub mean_pred: f64,
    pub roc_auc: Option<f64>,
    pub pr_auc: Option<f64>,
    pub brier_score: f64,
    pub ece: f64,
    pub tpr: f64,
    pub fpr: f64,
    pub positive_pred_rate: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct FairnessSummary {
    pub group_col: String,
    pub threshold: f64,
    pub max_tpr_gap: f64,
    pub max_ece_gap: f64,
    pub max_auc_gap: f64,
    pub groups: Vec<SubgroupBinaryMetric>,
}

#[derive(Debug, Clone, Serialize)]
pub struct BinaryEvalReport {
    pub tool: String,
    pub version: String,
    pub task: String,
    pub title: String,
    pub generated_at: String,
    pub source: String,
    pub n_rows: usize,
    pub positives: usize,
    pub base_rate: f64,
    pub mean_predicted: f64,
    pub roc_auc: f64,
    pub pr_auc: f64,
    pub brier_score: f64,
    pub brier_skill_score: f64,
    pub log_loss: f64,
    pub ece: f64,
    pub threshold: f64,
    pub precision_at_threshold: f64,
    pub recall_at_threshold: f64,
    pub f1_at_threshold: f64,
    pub top_k: Vec<TopKMetric>,
    pub calibration_bins: Vec<CalibrationBin>,
    pub comparison: Option<BinaryCompare>,
    pub fairness: Option<FairnessSummary>,
    pub folds: Option<Vec<SubgroupBinaryMetric>>,
    pub gates: Vec<GateResult>,
    pub gates_passed: bool,
    pub sha256: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct ConformalSummary {
    pub p10_col: String,
    pub p90_col: String,
    pub empirical_coverage: f64,
    pub below_p10_rate: f64,
    pub above_p90_rate: f64,
    pub mean_interval_width: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct CapacitySummary {
    pub cap_per_section: f64,
    pub exact_section_accuracy: f64,
    pub within_one_section_accuracy: f64,
    pub under_staffed_rate: f64,
    pub over_staffed_rate: f64,
    pub total_actual_sections: usize,
    pub total_pred_sections: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct RegressCompare {
    pub baseline_col: String,
    pub baseline_mae: f64,
    pub baseline_rmse: f64,
    pub baseline_wmape: f64,
    pub delta_mae: f64,
    pub delta_wmape: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct SubgroupRegressMetric {
    pub group: String,
    pub n: usize,
    pub mae: f64,
    pub rmse: f64,
    pub wmape: f64,
    pub r2: f64,
    pub bias: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct RegressEvalReport {
    pub tool: String,
    pub version: String,
    pub task: String,
    pub title: String,
    pub generated_at: String,
    pub source: String,
    pub n_rows: usize,
    pub mean_actual: f64,
    pub mean_predicted: f64,
    pub mae: f64,
    pub rmse: f64,
    pub medae: f64,
    pub wmape: f64,
    pub r2: f64,
    pub bias: f64,
    pub conformal: Option<ConformalSummary>,
    pub capacity: Option<CapacitySummary>,
    pub comparison: Option<RegressCompare>,
    pub groups: Option<Vec<SubgroupRegressMetric>>,
    pub gates: Vec<GateResult>,
    pub gates_passed: bool,
    pub sha256: String,
}

/// Exact tie-aware ROC-AUC matching `sklearn.metrics.roc_auc_score`.
pub fn roc_auc(pairs: &[(f64, f64)]) -> Option<f64> {
    let total_pos: f64 = pairs.iter().map(|&(y, _)| y).sum();
    let total_neg: f64 = (pairs.len() as f64) - total_pos;
    if total_pos <= 0.0 || total_neg <= 0.0 {
        return None;
    }
    let mut sorted = pairs.to_vec();
    sorted.sort_by(|a, b| b.1.total_cmp(&a.1));

    let mut tp = 0.0_f64;
    let mut area = 0.0_f64;
    let mut i = 0usize;
    while i < sorted.len() {
        let score = sorted[i].1;
        let mut group_tp = 0.0_f64;
        let mut group_fp = 0.0_f64;
        while i < sorted.len() && sorted[i].1 == score {
            if sorted[i].0 >= 0.5 {
                group_tp += 1.0;
            } else {
                group_fp += 1.0;
            }
            i += 1;
        }
        // Trapezoidal area contributed by this tie group
        area += group_fp * (tp + 0.5 * group_tp);
        tp += group_tp;
    }
    Some(area / (total_pos * total_neg))
}

/// Exact step-function Average Precision (PR-AUC) matching `sklearn.metrics.average_precision_score`.
pub fn average_precision(pairs: &[(f64, f64)]) -> Option<f64> {
    let total_pos: f64 = pairs.iter().map(|&(y, _)| y).sum();
    if total_pos <= 0.0 {
        return None;
    }
    let mut sorted = pairs.to_vec();
    sorted.sort_by(|a, b| b.1.total_cmp(&a.1));

    let mut tp_cum = 0.0_f64;
    let mut n_cum = 0.0_f64;
    let mut ap = 0.0_f64;
    let mut i = 0usize;
    while i < sorted.len() {
        let score = sorted[i].1;
        let mut group_tp = 0.0_f64;
        let mut group_n = 0.0_f64;
        while i < sorted.len() && sorted[i].1 == score {
            if sorted[i].0 >= 0.5 {
                group_tp += 1.0;
            }
            group_n += 1.0;
            i += 1;
        }
        tp_cum += group_tp;
        n_cum += group_n;
        if group_tp > 0.0 {
            let delta_recall = group_tp / total_pos;
            let precision = tp_cum / n_cum;
            ap += delta_recall * precision;
        }
    }
    Some(ap)
}

pub fn calibration_curve(pairs: &[(f64, f64)]) -> (f64, Vec<CalibrationBin>) {
    let n = pairs.len().max(1) as f64;
    let mut cnt = [0usize; 10];
    let mut p_sum = [0.0_f64; 10];
    let mut y_sum = [0.0_f64; 10];
    for &(y, p) in pairs {
        let pc = p.clamp(0.0, 1.0);
        let b = ((pc * 10.0).floor() as usize).min(9);
        cnt[b] += 1;
        p_sum[b] += pc;
        y_sum[b] += y;
    }
    let mut ece = 0.0_f64;
    let mut bins = Vec::with_capacity(10);
    for b in 0..10 {
        let lo = (b as f64) * 0.1;
        let hi = ((b + 1) as f64) * 0.1;
        if cnt[b] > 0 {
            let c = cnt[b] as f64;
            let mp = p_sum[b] / c;
            let or = y_sum[b] / c;
            let gap = (mp - or).abs();
            ece += (c / n) * gap;
            bins.push(CalibrationBin {
                bin_index: b,
                range_label: format!("{lo:.1}-{hi:.1}"),
                count: cnt[b],
                mean_predicted: r6(mp),
                observed_rate: r6(or),
                abs_gap: r6(gap),
            });
        } else {
            bins.push(CalibrationBin {
                bin_index: b,
                range_label: format!("{lo:.1}-{hi:.1}"),
                count: 0,
                mean_predicted: r6((lo + hi) * 0.5),
                observed_rate: 0.0,
                abs_gap: 0.0,
            });
        }
    }
    (ece, bins)
}

fn eval_subgroup_binary(group: &str, pairs: &[(f64, f64)], threshold: f64) -> SubgroupBinaryMetric {
    let n = pairs.len();
    let nf = n.max(1) as f64;
    let positives = pairs.iter().filter(|&&(y, _)| y >= 0.5).count();
    let negatives = n.saturating_sub(positives);
    let base_rate = (positives as f64) / nf;
    let mean_pred = pairs.iter().map(|&(_, p)| p).sum::<f64>() / nf;
    let brier = pairs.iter().map(|&(y, p)| (p - y).powi(2)).sum::<f64>() / nf;
    let (ece, _) = calibration_curve(pairs);
    let mut tp = 0usize;
    let mut fp = 0usize;
    for &(y, p) in pairs {
        if p >= threshold {
            if y >= 0.5 {
                tp += 1;
            } else {
                fp += 1;
            }
        }
    }
    SubgroupBinaryMetric {
        group: group.to_string(),
        n,
        positives,
        base_rate: r4(base_rate),
        mean_pred: r4(mean_pred),
        roc_auc: roc_auc(pairs).map(r6),
        pr_auc: average_precision(pairs).map(r6),
        brier_score: r6(brier),
        ece: r6(ece),
        tpr: if positives > 0 {
            r4((tp as f64) / (positives as f64))
        } else {
            0.0
        },
        fpr: if negatives > 0 {
            r4((fp as f64) / (negatives as f64))
        } else {
            0.0
        },
        positive_pred_rate: r4(((tp + fp) as f64) / nf),
    }
}

#[allow(clippy::too_many_arguments)]
pub fn evaluate_binary(
    table: &DataTable,
    source: &str,
    y_col: &str,
    p_col: &str,
    compare_col: Option<&str>,
    group_col: Option<&str>,
    fold_col: Option<&str>,
    threshold: f64,
    top_k_pcts: &[f64],
    title: Option<&str>,
    min_pr_auc: Option<f64>,
    min_roc_auc: Option<f64>,
    max_brier: Option<f64>,
    max_ece: Option<f64>,
    max_group_tpr_gap: Option<f64>,
) -> Result<BinaryEvalReport, String> {
    let ys = table
        .columns
        .get(y_col)
        .ok_or_else(|| format!("label column `{y_col}` not found"))?;
    let ps = table
        .columns
        .get(p_col)
        .ok_or_else(|| format!("probability column `{p_col}` not found"))?;

    let mut pairs: Vec<(f64, f64)> = Vec::new();
    let mut row_indices: Vec<usize> = Vec::new();
    for idx in 0..table.row_count {
        if let (Some(ys_s), Some(ps_s)) = (&ys[idx], &ps[idx])
            && let (Ok(yv), Ok(pv)) = (ys_s.parse::<f64>(), ps_s.parse::<f64>())
            && yv.is_finite()
            && pv.is_finite()
        {
            pairs.push((if yv >= 0.5 { 1.0 } else { 0.0 }, pv.clamp(0.0, 1.0)));
            row_indices.push(idx);
        }
    }
    if pairs.is_empty() {
        return Err("no valid (y, p) pairs found".to_string());
    }

    let n = pairs.len();
    let nf = n as f64;
    let positives = pairs.iter().filter(|&&(y, _)| y >= 0.5).count();
    let base_rate = (positives as f64) / nf;
    let mean_predicted = pairs.iter().map(|&(_, p)| p).sum::<f64>() / nf;
    let auc = roc_auc(&pairs).unwrap_or(0.5);
    let ap = average_precision(&pairs).unwrap_or(base_rate);
    let brier = pairs.iter().map(|&(y, p)| (p - y).powi(2)).sum::<f64>() / nf;
    let brier_ref = base_rate * (1.0 - base_rate);
    let brier_skill = if brier_ref > 1e-12 {
        1.0 - brier / brier_ref
    } else {
        0.0
    };
    let log_loss = -pairs
        .iter()
        .map(|&(y, p)| {
            let pc = p.clamp(1e-15, 1.0 - 1e-15);
            y * pc.ln() + (1.0 - y) * (1.0 - pc).ln()
        })
        .sum::<f64>()
        / nf;
    let (ece, cal_bins) = calibration_curve(&pairs);

    // Threshold metrics
    let (mut tp, mut fp, mut fn_cnt) = (0usize, 0usize, 0usize);
    for &(y, p) in &pairs {
        let pred_pos = p >= threshold;
        let act_pos = y >= 0.5;
        match (pred_pos, act_pos) {
            (true, true) => tp += 1,
            (true, false) => fp += 1,
            (false, true) => fn_cnt += 1,
            (false, false) => {}
        }
    }
    let prec_t = if tp + fp > 0 {
        (tp as f64) / ((tp + fp) as f64)
    } else {
        0.0
    };
    let rec_t = if tp + fn_cnt > 0 {
        (tp as f64) / ((tp + fn_cnt) as f64)
    } else {
        0.0
    };
    let f1_t = if prec_t + rec_t > 0.0 {
        2.0 * prec_t * rec_t / (prec_t + rec_t)
    } else {
        0.0
    };

    // Top-K% capture
    let mut sorted_desc = pairs.clone();
    sorted_desc.sort_by(|a, b| b.1.total_cmp(&a.1));
    let mut top_k = Vec::new();
    for &k_pct in top_k_pcts {
        let k_clamped = k_pct.clamp(0.1, 100.0);
        let n_flag = ((nf * k_clamped / 100.0).ceil() as usize).clamp(1, n);
        let pos_cap = sorted_desc[..n_flag]
            .iter()
            .filter(|&&(y, _)| y >= 0.5)
            .count();
        let prec = (pos_cap as f64) / (n_flag as f64);
        let rec = if positives > 0 {
            (pos_cap as f64) / (positives as f64)
        } else {
            0.0
        };
        let lift = if base_rate > 1e-12 {
            prec / base_rate
        } else {
            0.0
        };
        top_k.push(TopKMetric {
            top_k_pct: k_clamped,
            n_flagged: n_flag,
            positives_captured: pos_cap,
            precision: r4(prec),
            recall: r4(rec),
            lift: r4(lift),
        });
    }

    // Optional baseline comparison
    let comparison = if let Some(bcol) = compare_col {
        let bs = table
            .columns
            .get(bcol)
            .ok_or_else(|| format!("compare column `{bcol}` not found"))?;
        let mut bpairs = Vec::new();
        for (&ridx, &(y, _)) in row_indices.iter().zip(pairs.iter()) {
            if let Some(s) = &bs[ridx]
                && let Ok(bv) = s.parse::<f64>()
                && bv.is_finite()
            {
                bpairs.push((y, bv.clamp(0.0, 1.0)));
            }
        }
        if !bpairs.is_empty() {
            let b_auc = roc_auc(&bpairs).unwrap_or(0.5);
            let b_ap = average_precision(&bpairs).unwrap_or(base_rate);
            let b_brier = bpairs.iter().map(|&(y, p)| (p - y).powi(2)).sum::<f64>()
                / (bpairs.len() as f64);
            Some(BinaryCompare {
                baseline_col: bcol.to_string(),
                baseline_roc_auc: r6(b_auc),
                baseline_pr_auc: r6(b_ap),
                baseline_brier: r6(b_brier),
                delta_roc_auc: r6(auc - b_auc),
                delta_pr_auc: r6(ap - b_ap),
                delta_brier: r6(brier - b_brier),
            })
        } else {
            None
        }
    } else {
        None
    };

    // Optional subgroup fairness parity
    let fairness = if let Some(gcol) = group_col {
        let gs = table
            .columns
            .get(gcol)
            .ok_or_else(|| format!("group column `{gcol}` not found"))?;
        let mut by_group: BTreeMap<String, Vec<(f64, f64)>> = BTreeMap::new();
        for (&ridx, &pair) in row_indices.iter().zip(pairs.iter()) {
            let gname = gs[ridx].clone().unwrap_or_else(|| "(missing)".to_string());
            by_group.entry(gname).or_default().push(pair);
        }
        let groups: Vec<SubgroupBinaryMetric> = by_group
            .iter()
            .map(|(g, gp)| eval_subgroup_binary(g, gp, threshold))
            .collect();
        let tprs: Vec<f64> = groups.iter().filter(|g| g.positives > 0).map(|g| g.tpr).collect();
        let eces: Vec<f64> = groups.iter().map(|g| g.ece).collect();
        let aucs: Vec<f64> = groups.iter().filter_map(|g| g.roc_auc).collect();
        let span = |vals: &[f64]| -> f64 {
            if vals.len() < 2 {
                0.0
            } else {
                let mn = vals.iter().copied().fold(f64::INFINITY, f64::min);
                let mx = vals.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                r4(mx - mn)
            }
        };
        Some(FairnessSummary {
            group_col: gcol.to_string(),
            threshold,
            max_tpr_gap: span(&tprs),
            max_ece_gap: span(&eces),
            max_auc_gap: span(&aucs),
            groups,
        })
    } else {
        None
    };

    // Optional fold breakdown
    let folds = if let Some(fcol) = fold_col {
        let fs_col = table
            .columns
            .get(fcol)
            .ok_or_else(|| format!("fold column `{fcol}` not found"))?;
        let mut by_fold: BTreeMap<String, Vec<(f64, f64)>> = BTreeMap::new();
        for (&ridx, &pair) in row_indices.iter().zip(pairs.iter()) {
            let fname = fs_col[ridx].clone().unwrap_or_else(|| "(missing)".to_string());
            by_fold.entry(fname).or_default().push(pair);
        }
        Some(
            by_fold
                .iter()
                .map(|(f, fp)| eval_subgroup_binary(f, fp, threshold))
                .collect(),
        )
    } else {
        None
    };

    // CI Quality Gates
    let mut gates = Vec::new();
    if let Some(min_v) = min_pr_auc {
        gates.push(GateResult {
            metric: "pr_auc".to_string(),
            rule: format!(">= {min_v:.4}"),
            actual: r6(ap),
            passed: ap >= min_v,
        });
    }
    if let Some(min_v) = min_roc_auc {
        gates.push(GateResult {
            metric: "roc_auc".to_string(),
            rule: format!(">= {min_v:.4}"),
            actual: r6(auc),
            passed: auc >= min_v,
        });
    }
    if let Some(max_v) = max_brier {
        gates.push(GateResult {
            metric: "brier_score".to_string(),
            rule: format!("<= {max_v:.4}"),
            actual: r6(brier),
            passed: brier <= max_v,
        });
    }
    if let Some(max_v) = max_ece {
        gates.push(GateResult {
            metric: "ece".to_string(),
            rule: format!("<= {max_v:.4}"),
            actual: r6(ece),
            passed: ece <= max_v,
        });
    }
    if let (Some(max_gap), Some(fair)) = (max_group_tpr_gap, &fairness) {
        gates.push(GateResult {
            metric: "max_group_tpr_gap".to_string(),
            rule: format!("<= {max_gap:.4}"),
            actual: fair.max_tpr_gap,
            passed: fair.max_tpr_gap <= max_gap,
        });
    }
    let gates_passed = gates.iter().all(|g| g.passed);

    let mut hasher = Sha256::new();
    hasher.update(
        format!(
            "binary:{source}:{n}:{positives}:{auc:.6}:{ap:.6}:{brier:.6}:{ece:.6}"
        )
        .as_bytes(),
    );
    let sha256: String = hasher.finalize().iter().map(|b| format!("{b:02x}")).collect();

    Ok(BinaryEvalReport {
        tool: "evalkeel".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        task: "binary".to_string(),
        title: title
            .map(|s| s.to_string())
            .unwrap_or_else(|| format!("Binary Classification Model Card ({p_col} vs {y_col})")),
        generated_at: Utc::now().to_rfc3339(),
        source: source.to_string(),
        n_rows: n,
        positives,
        base_rate: r6(base_rate),
        mean_predicted: r6(mean_predicted),
        roc_auc: r6(auc),
        pr_auc: r6(ap),
        brier_score: r6(brier),
        brier_skill_score: r6(brier_skill),
        log_loss: r6(log_loss),
        ece: r6(ece),
        threshold,
        precision_at_threshold: r4(prec_t),
        recall_at_threshold: r4(rec_t),
        f1_at_threshold: r4(f1_t),
        top_k,
        calibration_bins: cal_bins,
        comparison,
        fairness,
        folds,
        gates,
        gates_passed,
        sha256,
    })
}

fn reg_stats(pairs: &[(f64, f64)]) -> (f64, f64, f64, f64, f64, f64) {
    let n = pairs.len().max(1) as f64;
    let mean_y = pairs.iter().map(|&(y, _)| y).sum::<f64>() / n;
    let mut abs_errs: Vec<f64> = pairs.iter().map(|&(y, p)| (y - p).abs()).collect();
    let mae = abs_errs.iter().sum::<f64>() / n;
    let rmse = (pairs.iter().map(|&(y, p)| (y - p).powi(2)).sum::<f64>() / n).sqrt();
    abs_errs.sort_by(|a, b| a.total_cmp(b));
    let medae = if abs_errs.is_empty() {
        0.0
    } else if abs_errs.len() % 2 == 1 {
        abs_errs[abs_errs.len() / 2]
    } else {
        0.5 * (abs_errs[abs_errs.len() / 2 - 1] + abs_errs[abs_errs.len() / 2])
    };
    let sum_abs_y: f64 = pairs.iter().map(|&(y, _)| y.abs()).sum();
    let wmape = if sum_abs_y > 1e-12 {
        abs_errs.iter().sum::<f64>() / sum_abs_y
    } else {
        0.0
    };
    let ss_res: f64 = pairs.iter().map(|&(y, p)| (y - p).powi(2)).sum();
    let ss_tot: f64 = pairs.iter().map(|&(y, _)| (y - mean_y).powi(2)).sum();
    let r2 = if ss_tot > 1e-12 {
        1.0 - ss_res / ss_tot
    } else {
        0.0
    };
    let bias = pairs.iter().map(|&(y, p)| p - y).sum::<f64>() / n;
    (mae, rmse, medae, wmape, r2, bias)
}

#[allow(clippy::too_many_arguments)]
pub fn evaluate_regress(
    table: &DataTable,
    source: &str,
    y_col: &str,
    pred_col: &str,
    p10_col: Option<&str>,
    p90_col: Option<&str>,
    compare_col: Option<&str>,
    group_col: Option<&str>,
    cap: Option<f64>,
    title: Option<&str>,
    max_mae: Option<f64>,
    max_wmape: Option<f64>,
    min_r2: Option<f64>,
    min_coverage: Option<f64>,
) -> Result<RegressEvalReport, String> {
    let ys = table
        .columns
        .get(y_col)
        .ok_or_else(|| format!("target column `{y_col}` not found"))?;
    let preds = table
        .columns
        .get(pred_col)
        .ok_or_else(|| format!("prediction column `{pred_col}` not found"))?;

    let mut pairs: Vec<(f64, f64)> = Vec::new();
    let mut row_indices: Vec<usize> = Vec::new();
    for idx in 0..table.row_count {
        if let (Some(ys_s), Some(ps_s)) = (&ys[idx], &preds[idx])
            && let (Ok(yv), Ok(pv)) = (ys_s.parse::<f64>(), ps_s.parse::<f64>())
            && yv.is_finite()
            && pv.is_finite()
        {
            pairs.push((yv, pv));
            row_indices.push(idx);
        }
    }
    if pairs.is_empty() {
        return Err("no valid (y, pred) pairs found".to_string());
    }

    let n = pairs.len();
    let nf = n as f64;
    let mean_actual = pairs.iter().map(|&(y, _)| y).sum::<f64>() / nf;
    let mean_predicted = pairs.iter().map(|&(_, p)| p).sum::<f64>() / nf;
    let (mae, rmse, medae, wmape, r2, bias) = reg_stats(&pairs);

    // Optional Conformal P10-P90 interval coverage
    let conformal = if let (Some(c10), Some(c90)) = (p10_col, p90_col) {
        let lo_col = table
            .columns
            .get(c10)
            .ok_or_else(|| format!("p10 column `{c10}` not found"))?;
        let hi_col = table
            .columns
            .get(c90)
            .ok_or_else(|| format!("p90 column `{c90}` not found"))?;
        let (mut inside, mut below, mut above, mut valid) = (0usize, 0usize, 0usize, 0usize);
        let mut width_sum = 0.0_f64;
        for (&ridx, &(y, _)) in row_indices.iter().zip(pairs.iter()) {
            if let (Some(ls), Some(hs)) = (&lo_col[ridx], &hi_col[ridx])
                && let (Ok(lo), Ok(hi)) = (ls.parse::<f64>(), hs.parse::<f64>())
                && lo.is_finite()
                && hi.is_finite()
            {
                valid += 1;
                width_sum += (hi - lo).max(0.0);
                if y < lo {
                    below += 1;
                } else if y > hi {
                    above += 1;
                } else {
                    inside += 1;
                }
            }
        }
        if valid > 0 {
            let vf = valid as f64;
            Some(ConformalSummary {
                p10_col: c10.to_string(),
                p90_col: c90.to_string(),
                empirical_coverage: r4((inside as f64) / vf),
                below_p10_rate: r4((below as f64) / vf),
                above_p90_rate: r4((above as f64) / vf),
                mean_interval_width: r4(width_sum / vf),
            })
        } else {
            None
        }
    } else {
        None
    };

    // Optional section/capacity staffing accuracy
    let capacity = cap.filter(|&c| c > 0.0).map(|cap_val| {
        let sections = |v: f64| -> usize {
            if v <= 0.0 {
                0
            } else {
                (v / cap_val).ceil() as usize
            }
        };
        let (mut exact, mut within1, mut under, mut over) = (0usize, 0usize, 0usize, 0usize);
        let (mut tot_act, mut tot_pred) = (0usize, 0usize);
        for &(y, p) in &pairs {
            let sa = sections(y);
            let sp = sections(p);
            tot_act += sa;
            tot_pred += sp;
            if sa == sp {
                exact += 1;
            }
            if sa.abs_diff(sp) <= 1 {
                within1 += 1;
            }
            if sp < sa {
                under += 1;
            } else if sp > sa {
                over += 1;
            }
        }
        CapacitySummary {
            cap_per_section: cap_val,
            exact_section_accuracy: r4((exact as f64) / nf),
            within_one_section_accuracy: r4((within1 as f64) / nf),
            under_staffed_rate: r4((under as f64) / nf),
            over_staffed_rate: r4((over as f64) / nf),
            total_actual_sections: tot_act,
            total_pred_sections: tot_pred,
        }
    });

    // Optional baseline comparison
    let comparison = if let Some(bcol) = compare_col {
        let bs = table
            .columns
            .get(bcol)
            .ok_or_else(|| format!("compare column `{bcol}` not found"))?;
        let mut bpairs = Vec::new();
        for (&ridx, &(y, _)) in row_indices.iter().zip(pairs.iter()) {
            if let Some(s) = &bs[ridx]
                && let Ok(bv) = s.parse::<f64>()
                && bv.is_finite()
            {
                bpairs.push((y, bv));
            }
        }
        if !bpairs.is_empty() {
            let (b_mae, b_rmse, _, b_wmape, _, _) = reg_stats(&bpairs);
            Some(RegressCompare {
                baseline_col: bcol.to_string(),
                baseline_mae: r4(b_mae),
                baseline_rmse: r4(b_rmse),
                baseline_wmape: r4(b_wmape),
                delta_mae: r4(mae - b_mae),
                delta_wmape: r4(wmape - b_wmape),
            })
        } else {
            None
        }
    } else {
        None
    };

    // Optional subgroup breakdown
    let groups = if let Some(gcol) = group_col {
        let gs = table
            .columns
            .get(gcol)
            .ok_or_else(|| format!("group column `{gcol}` not found"))?;
        let mut by_g: BTreeMap<String, Vec<(f64, f64)>> = BTreeMap::new();
        for (&ridx, &pair) in row_indices.iter().zip(pairs.iter()) {
            let gname = gs[ridx].clone().unwrap_or_else(|| "(missing)".to_string());
            by_g.entry(gname).or_default().push(pair);
        }
        Some(
            by_g.iter()
                .map(|(g, gp)| {
                    let (gmae, grmse, _, gwmape, gr2, gbias) = reg_stats(gp);
                    SubgroupRegressMetric {
                        group: g.clone(),
                        n: gp.len(),
                        mae: r4(gmae),
                        rmse: r4(grmse),
                        wmape: r4(gwmape),
                        r2: r4(gr2),
                        bias: r4(gbias),
                    }
                })
                .collect(),
        )
    } else {
        None
    };

    // CI Quality Gates
    let mut gates = Vec::new();
    if let Some(mx) = max_mae {
        gates.push(GateResult {
            metric: "mae".to_string(),
            rule: format!("<= {mx:.4}"),
            actual: r4(mae),
            passed: mae <= mx,
        });
    }
    if let Some(mx) = max_wmape {
        gates.push(GateResult {
            metric: "wmape".to_string(),
            rule: format!("<= {mx:.4}"),
            actual: r4(wmape),
            passed: wmape <= mx,
        });
    }
    if let Some(mn) = min_r2 {
        gates.push(GateResult {
            metric: "r2".to_string(),
            rule: format!(">= {mn:.4}"),
            actual: r4(r2),
            passed: r2 >= mn,
        });
    }
    if let (Some(mn), Some(conf)) = (min_coverage, &conformal) {
        gates.push(GateResult {
            metric: "conformal_coverage".to_string(),
            rule: format!(">= {mn:.4}"),
            actual: conf.empirical_coverage,
            passed: conf.empirical_coverage >= mn,
        });
    }
    let gates_passed = gates.iter().all(|g| g.passed);

    let mut hasher = Sha256::new();
    hasher.update(format!("regress:{source}:{n}:{mae:.6}:{rmse:.6}:{wmape:.6}:{r2:.6}").as_bytes());
    let sha256: String = hasher.finalize().iter().map(|b| format!("{b:02x}")).collect();

    Ok(RegressEvalReport {
        tool: "evalkeel".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        task: "regress".to_string(),
        title: title
            .map(|s| s.to_string())
            .unwrap_or_else(|| format!("Regression & Capacity Model Card ({pred_col} vs {y_col})")),
        generated_at: Utc::now().to_rfc3339(),
        source: source.to_string(),
        n_rows: n,
        mean_actual: r4(mean_actual),
        mean_predicted: r4(mean_predicted),
        mae: r4(mae),
        rmse: r4(rmse),
        medae: r4(medae),
        wmape: r4(wmape),
        r2: r6(r2),
        bias: r4(bias),
        conformal,
        capacity,
        comparison,
        groups,
        gates,
        gates_passed,
        sha256,
    })
}
