use std::fs;

use assert_cmd::Command;
use tempfile::tempdir;

#[test]
fn binary_eval_fairness_comparison_and_gates() {
    let dir = tempdir().unwrap();
    let csv_path = dir.path().join("preds.csv");
    let html_path = dir.path().join("card.html");

    let mut content = String::from("y,p,base_p,program\n");
    // Construct clean discriminative probabilities across 2 groups
    for i in 0..100 {
        let prog = if i % 2 == 0 { "MDiv" } else { "MAC" };
        let (y, p, bp) = if i < 25 {
            (1, 0.80 - (i as f64) * 0.005, 0.25)
        } else {
            (0, 0.10 + ((i - 25) as f64) * 0.004, 0.25)
        };
        content.push_str(&format!("{y},{p:.4},{bp:.4},{prog}\n"));
    }
    fs::write(&csv_path, content).unwrap();

    let out = Command::cargo_bin("evalkeel")
        .unwrap()
        .args([
            "binary",
            csv_path.to_str().unwrap(),
            "--y",
            "y",
            "--p",
            "p",
            "--compare",
            "base_p",
            "--group",
            "program",
            "--min-pr-auc",
            "0.95",
            "--min-roc-auc",
            "0.95",
            "--html",
            html_path.to_str().unwrap(),
            "--format",
            "agent",
        ])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();

    let parsed: serde_json::Value = serde_json::from_slice(&out).unwrap();
    assert_eq!(parsed["gates_passed"], true);
    assert!((parsed["roc_auc"].as_f64().unwrap() - 1.0).abs() < 1e-6);
    assert!((parsed["pr_auc"].as_f64().unwrap() - 1.0).abs() < 1e-6);
    assert!(parsed["comparison"]["delta_pr_auc"].as_f64().unwrap() > 0.0);
    assert_eq!(parsed["fairness"]["groups"].as_array().unwrap().len(), 2);
    assert!(fs::read_to_string(&html_path).unwrap().contains("<svg"));

    // Gate failure must exit 1
    Command::cargo_bin("evalkeel")
        .unwrap()
        .args([
            "binary",
            csv_path.to_str().unwrap(),
            "--y",
            "y",
            "--p",
            "p",
            "--max-brier",
            "0.001",
            "--format",
            "agent",
        ])
        .assert()
        .code(1);
}

#[test]
fn regress_conformal_and_capacity_eval() {
    let dir = tempdir().unwrap();
    let csv_path = dir.path().join("demand.csv");
    let html_path = dir.path().join("demand_card.html");
    let content = "\
actual,p50,p10,p90,naive,dept
24,23,18,29,18,OT
52,50,42,60,40,NT
18,19,14,24,25,ST
31,30,24,38,22,PT
45,44,36,52,35,CH
";
    fs::write(&csv_path, content).unwrap();

    let out = Command::cargo_bin("evalkeel")
        .unwrap()
        .args([
            "regress",
            csv_path.to_str().unwrap(),
            "--y",
            "actual",
            "--pred",
            "p50",
            "--p10",
            "p10",
            "--p90",
            "p90",
            "--cap",
            "25",
            "--compare",
            "naive",
            "--group",
            "dept",
            "--max-wmape",
            "0.10",
            "--min-coverage",
            "0.80",
            "--html",
            html_path.to_str().unwrap(),
            "--format",
            "agent",
        ])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();

    let parsed: serde_json::Value = serde_json::from_slice(&out).unwrap();
    assert_eq!(parsed["gates_passed"], true);
    assert_eq!(parsed["conformal"]["empirical_coverage"], 1.0);
    // 52 actual -> 3 sections vs 50 pred -> 2 sections (4/5 exact, 5/5 within +/-1, 1/5 under-staffed)
    assert_eq!(parsed["capacity"]["exact_section_accuracy"], 0.8);
    assert_eq!(parsed["capacity"]["within_one_section_accuracy"], 1.0);
    assert_eq!(parsed["capacity"]["under_staffed_rate"], 0.2);
    assert!(parsed["comparison"]["delta_wmape"].as_f64().unwrap() < 0.0);
    assert!(fs::read_to_string(&html_path).unwrap().contains("Capacity"));
}

#[test]
fn skill_prints_and_installs() {
    let dir = tempdir().unwrap();
    Command::cargo_bin("evalkeel")
        .unwrap()
        .args(["skill", "--install", dir.path().to_str().unwrap(), "--format", "human"])
        .assert()
        .success();
    assert!(dir.path().join("evalkeel/SKILL.md").exists());
}
