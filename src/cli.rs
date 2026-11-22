use std::io::IsTerminal;
use std::path::PathBuf;

use clap::{Parser, Subcommand, ValueEnum};

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum OutputFormat {
    Auto,
    Human,
    Json,
    Agent,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResolvedFormat {
    Human,
    Json,
    Agent,
}

impl OutputFormat {
    pub fn resolve(self) -> ResolvedFormat {
        match self {
            Self::Human => ResolvedFormat::Human,
            Self::Json => ResolvedFormat::Json,
            Self::Agent => ResolvedFormat::Agent,
            Self::Auto => {
                if std::io::stdout().is_terminal() {
                    ResolvedFormat::Human
                } else {
                    ResolvedFormat::Json
                }
            }
        }
    }
}

#[derive(Debug, Parser)]
#[command(
    name = "evalkeel",
    version,
    about = "Zero-Python Rust model-agnostic ML backtest, calibration, conformal coverage, fairness parity & HTML Model Card engine."
)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Command,
}

#[derive(Debug, Subcommand)]
pub enum Command {
    /// Evaluate binary classification predictions (ROC-AUC, PR-AUC, Brier, ECE, Top-K Lift, fairness parity)
    Binary {
        /// Predictions dataset (.csv, .tsv, .jsonl, .ndjson, .parquet)
        input: PathBuf,
        /// Ground-truth binary label column (0/1)
        #[arg(long)]
        y: String,
        /// Predicted probability column
        #[arg(long)]
        p: String,
        /// Optional baseline probability column for head-to-head delta
        #[arg(long)]
        compare: Option<String>,
        /// Optional subgroup column for fairness & calibration parity slicing
        #[arg(long)]
        group: Option<String>,
        /// Optional out-of-time fold/term column for temporal stability slicing
        #[arg(long)]
        fold: Option<String>,
        /// Decision threshold for TPR/FPR/F1 classification metrics
        #[arg(long, default_value_t = 0.5)]
        threshold: f64,
        /// Top-K% capture cutoffs (comma-separated)
        #[arg(long, value_delimiter = ',', default_values_t = vec![5.0, 10.0, 20.0])]
        top_k: Vec<f64>,
        /// Optional model display title for the HTML Model Card
        #[arg(long)]
        title: Option<String>,
        /// CI gate: minimum required PR-AUC (Average Precision)
        #[arg(long)]
        min_pr_auc: Option<f64>,
        /// CI gate: minimum required ROC-AUC
        #[arg(long)]
        min_roc_auc: Option<f64>,
        /// CI gate: maximum allowed Brier score
        #[arg(long)]
        max_brier: Option<f64>,
        /// CI gate: maximum allowed Expected Calibration Error (ECE)
        #[arg(long)]
        max_ece: Option<f64>,
        /// CI gate: maximum allowed subgroup TPR parity gap
        #[arg(long)]
        max_group_tpr_gap: Option<f64>,
        /// Write tamper-evident zero-CDN offline HTML Model Card
        #[arg(long)]
        html: Option<PathBuf>,
        /// Output format (auto, human, json, agent)
        #[arg(long, value_enum, default_value_t = OutputFormat::Auto)]
        format: OutputFormat,
    },

    /// Evaluate regression, demand & conformal quantile predictions (MAE, RMSE, WMAPE, R2, P10-P90 coverage, capacity staffing)
    Regress {
        /// Predictions dataset (.csv, .tsv, .jsonl, .ndjson, .parquet)
        input: PathBuf,
        /// Ground-truth numeric target column
        #[arg(long)]
        y: String,
        /// Point prediction column (e.g. P50 / expected value)
        #[arg(long)]
        pred: String,
        /// Optional lower quantile / conformal bound column (e.g. P10)
        #[arg(long)]
        p10: Option<String>,
        /// Optional upper quantile / conformal bound column (e.g. P90)
        #[arg(long)]
        p90: Option<String>,
        /// Optional baseline prediction column for head-to-head delta
        #[arg(long)]
        compare: Option<String>,
        /// Optional subgroup column (e.g. department or term)
        #[arg(long)]
        group: Option<String>,
        /// Optional capacity per section/unit (e.g. 25 seats/section) for operational staffing accuracy
        #[arg(long)]
        cap: Option<f64>,
        /// Optional model display title for the HTML Model Card
        #[arg(long)]
        title: Option<String>,
        /// CI gate: maximum allowed MAE
        #[arg(long)]
        max_mae: Option<f64>,
        /// CI gate: maximum allowed WMAPE (e.g. 0.20 for 20%)
        #[arg(long)]
        max_wmape: Option<f64>,
        /// CI gate: minimum required R2
        #[arg(long)]
        min_r2: Option<f64>,
        /// CI gate: minimum required P10-P90 empirical coverage (e.g. 0.72)
        #[arg(long)]
        min_coverage: Option<f64>,
        /// Write tamper-evident zero-CDN offline HTML Model Card
        #[arg(long)]
        html: Option<PathBuf>,
        /// Output format (auto, human, json, agent)
        #[arg(long, value_enum, default_value_t = OutputFormat::Auto)]
        format: OutputFormat,
    },

    /// Print or install the bundled SKILL.md for AI coding agents
    Skill {
        /// Install SKILL.md into <dir>/evalkeel/SKILL.md (defaults to .agents/skills)
        #[arg(long, num_args = 0..=1, default_missing_value = ".agents/skills")]
        install: Option<PathBuf>,
        /// Output format (kept for Keel stack CLI consistency)
        #[arg(long, value_enum, default_value_t = OutputFormat::Auto)]
        format: OutputFormat,
    },
}
