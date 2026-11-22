//! `evalkeel` — Model-agnostic ML backtest, calibration, conformal & fairness receipt engine.

mod cli;
mod html;
mod io_data;
mod metrics;
mod skill;

use std::fs;
use std::process::ExitCode;

use clap::Parser;
use cli::{Cli, Command, ResolvedFormat};
use io_data::read_table;
use metrics::{evaluate_binary, evaluate_regress};
use owo_colors::OwoColorize;
use sha2::{Digest, Sha256};

fn hash_file_sha256(path: &std::path::Path) -> Result<String, String> {
    let bytes =
        fs::read(path).map_err(|e| format!("failed reading {}: {e}", path.display()))?;
    let mut h = Sha256::new();
    h.update(&bytes);
    Ok(h.finalize().iter().map(|b| format!("{b:02x}")).collect())
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    match dispatch(cli) {
        Ok(code) => ExitCode::from(code),
        Err(err) => {
            eprintln!("error: {err}");
            ExitCode::from(2)
        }
    }
}

fn write_html_file(path: &std::path::Path, content: &str) -> Result<(), String> {
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent).map_err(|e| format!("failed creating dir: {e}"))?;
    }
    fs::write(path, content).map_err(|e| format!("failed writing {}: {e}", path.display()))?;
    eprintln!("wrote HTML Model Card -> {}", path.display());
    Ok(())
}

fn dispatch(cli: Cli) -> Result<u8, String> {
    match cli.command {
        Command::Skill { install, format: _ } => {
            skill::handle_skill(install.as_deref())?;
            Ok(0)
        }
        Command::Binary {
            input,
            y,
            p,
            compare,
            group,
            fold,
            threshold,
            top_k,
            title,
            min_pr_auc,
            min_roc_auc,
            max_brier,
            max_ece,
            max_group_tpr_gap,
            html: html_path,
            format,
        } => {
            let input_sha = hash_file_sha256(&input)?;
            let table = read_table(&input)?;
            let report = evaluate_binary(
                &table,
                &input.display().to_string(),
                &input_sha,
                &y,
                &p,
                compare.as_deref(),
                group.as_deref(),
                fold.as_deref(),
                threshold,
                &top_k,
                title.as_deref(),
                min_pr_auc,
                min_roc_auc,
                max_brier,
                max_ece,
                max_group_tpr_gap,
            )?;

            if let Some(hp) = &html_path {
                write_html_file(hp, &html::render_binary_html(&report))?;
            }

            match format.resolve() {
                ResolvedFormat::Json => println!("{}", serde_json::to_string_pretty(&report).unwrap()),
                ResolvedFormat::Agent => println!("{}", serde_json::to_string(&report).unwrap()),
                ResolvedFormat::Human => {
                    let status = if report.gates_passed {
                        "PASS".green().bold().to_string()
                    } else {
                        "FAIL".red().bold().to_string()
                    };
                    println!(
                        "{} binary {} (n={}, pos={}, base={:.1}%) | {}",
                        "evalkeel".bold(),
                        input.display(),
                        report.n_rows,
                        report.positives,
                        report.base_rate * 100.0,
                        status
                    );
                    println!(
                        "  PR-AUC={:.4}  ROC-AUC={:.4}  Brier={:.4} (BSS={:+.3})  ECE={:.4}  LogLoss={:.4}",
                        report.pr_auc,
                        report.roc_auc,
                        report.brier_score,
                        report.brier_skill_score,
                        report.ece,
                        report.log_loss
                    );
                    for tk in &report.top_k {
                        println!(
                            "  Top {:>4.0}% (n={:<4}): prec={:.1}%  recall={:.1}%  lift={:.2}x",
                            tk.top_k_pct,
                            tk.n_flagged,
                            tk.precision * 100.0,
                            tk.recall * 100.0,
                            tk.lift
                        );
                    }
                    if let Some(c) = &report.comparison {
                        println!(
                            "  vs {}: ΔPR-AUC={:+.4}  ΔROC-AUC={:+.4}  ΔBrier={:+.4}",
                            c.baseline_col, c.delta_pr_auc, c.delta_roc_auc, c.delta_brier
                        );
                    }
                    if let Some(f) = &report.fairness {
                        println!(
                            "  Fairness ({}): max_tpr_gap={:.1}pp  max_ece_gap={:.4}  max_auc_gap={:.4}",
                            f.group_col,
                            f.max_tpr_gap * 100.0,
                            f.max_ece_gap,
                            f.max_auc_gap
                        );
                    }
                }
            }
            Ok(if report.gates_passed { 0 } else { 1 })
        }
        Command::Regress {
            input,
            y,
            pred,
            p10,
            p90,
            compare,
            group,
            cap,
            title,
            max_mae,
            max_wmape,
            min_r2,
            min_coverage,
            html: html_path,
            format,
        } => {
            let input_sha = hash_file_sha256(&input)?;
            let table = read_table(&input)?;
            let report = evaluate_regress(
                &table,
                &input.display().to_string(),
                &input_sha,
                &y,
                &pred,
                p10.as_deref(),
                p90.as_deref(),
                compare.as_deref(),
                group.as_deref(),
                cap,
                title.as_deref(),
                max_mae,
                max_wmape,
                min_r2,
                min_coverage,
            )?;

            if let Some(hp) = &html_path {
                write_html_file(hp, &html::render_regress_html(&report))?;
            }

            match format.resolve() {
                ResolvedFormat::Json => println!("{}", serde_json::to_string_pretty(&report).unwrap()),
                ResolvedFormat::Agent => println!("{}", serde_json::to_string(&report).unwrap()),
                ResolvedFormat::Human => {
                    let status = if report.gates_passed {
                        "PASS".green().bold().to_string()
                    } else {
                        "FAIL".red().bold().to_string()
                    };
                    println!(
                        "{} regress {} (n={}) | {}",
                        "evalkeel".bold(),
                        input.display(),
                        report.n_rows,
                        status
                    );
                    println!(
                        "  WMAPE={:.1}%  MAE={:.2}  RMSE={:.2}  MedAE={:.2}  R²={:.4}  Bias={:+.2}",
                        report.wmape * 100.0,
                        report.mae,
                        report.rmse,
                        report.medae,
                        report.r2,
                        report.bias
                    );
                    if let Some(cf) = &report.conformal {
                        println!(
                            "  Conformal ({}–{}): coverage={:.1}%  mean_width={:.2}",
                            cf.p10_col,
                            cf.p90_col,
                            cf.empirical_coverage * 100.0,
                            cf.mean_interval_width
                        );
                    }
                    if let Some(cp) = &report.capacity {
                        println!(
                            "  Sections (cap={:.0}): exact={:.1}%  within±1={:.1}%  under={:.1}%",
                            cp.cap_per_section,
                            cp.exact_section_accuracy * 100.0,
                            cp.within_one_section_accuracy * 100.0,
                            cp.under_staffed_rate * 100.0
                        );
                    }
                }
            }
            Ok(if report.gates_passed { 0 } else { 1 })
        }
    }
}
