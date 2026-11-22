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
fn tie_aware_topk_strict_gates_crossing_quantiles_unclamped_logits_and_full_digest() {
    let dir = tempdir().unwrap();

    // 1. Constant predictor base_p=0.25 across 100 rows -> Top-20% precision=0.25, lift=1.0!
    let tie_csv = dir.path().join("ties.csv");
    let mut content = String::from("y,base_p,region\n");
    for i in 0..100 {
        let y = if i < 25 { 1 } else { 0 };
        let reg = if i % 2 == 0 { "NA" } else { "EU" };
        content.push_str(&format!("{y},0.25,{reg}\n"));
    }
    fs::write(&tie_csv, content).unwrap();
    let out = Command::cargo_bin("evalkeel")
        .unwrap()
        .args([
            "binary",
            tie_csv.to_str().unwrap(),
            "--y",
            "y",
            "--p",
            "base_p",
            "--group",
            "region",
            "--format",
            "agent",
        ])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let j: serde_json::Value = serde_json::from_slice(&out).unwrap();
    for tk in j["top_k"].as_array().unwrap() {
        assert!((tk["precision"].as_f64().unwrap() - 0.25).abs() < 1e-4);
        assert!((tk["lift"].as_f64().unwrap() - 1.0).abs() < 1e-4);
    }
    // "NA" preserved (not turned into "(missing)")
    let gnames: Vec<&str> = j["fairness"]["groups"]
        .as_array()
        .unwrap()
        .iter()
        .map(|g| g["group"].as_str().unwrap())
        .collect();
    assert!(gnames.contains(&"NA"));

    // 2. Unclamped negative log-odds give true ROC-AUC = 1.0
    let logit_csv = dir.path().join("logits.csv");
    fs::write(&logit_csv, "y,p\n1,-0.5\n0,-3.0\n").unwrap();
    let lout = Command::cargo_bin("evalkeel")
        .unwrap()
        .args([
            "binary",
            logit_csv.to_str().unwrap(),
            "--y",
            "y",
            "--p",
            "p",
            "--format",
            "agent",
        ])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let lj: serde_json::Value = serde_json::from_slice(&lout).unwrap();
    assert!((lj["roc_auc"].as_f64().unwrap() - 1.0).abs() < 1e-6);

    // 3. Crossing quantiles (p10=40, p90=20, y=35) + --cap overflow safety + --min-coverage without p10/p90
    let cross_csv = dir.path().join("cross.csv");
    fs::write(&cross_csv, "actual,p50,p10,p90\n35,30,40,20\n").unwrap();
    let cout = Command::cargo_bin("evalkeel")
        .unwrap()
        .args([
            "regress",
            cross_csv.to_str().unwrap(),
            "--y",
            "actual",
            "--pred",
            "p50",
            "--p10",
            "p10",
            "--p90",
            "p90",
            "--format",
            "agent",
        ])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let cj: serde_json::Value = serde_json::from_slice(&cout).unwrap();
    assert_eq!(cj["conformal"]["crossing_rate"], 1.0);
    assert_eq!(cj["conformal"]["below_p10_rate"], 1.0);
    assert_eq!(cj["conformal"]["above_p90_rate"], 1.0);
    assert_eq!(cj["conformal"]["empirical_coverage"], 0.0);

    // Gate without required columns exits 2
    Command::cargo_bin("evalkeel")
        .unwrap()
        .args([
            "regress",
            cross_csv.to_str().unwrap(),
            "--y",
            "actual",
            "--pred",
            "p50",
            "--min-coverage",
            "0.8",
        ])
        .assert()
        .code(2);
    Command::cargo_bin("evalkeel")
        .unwrap()
        .args([
            "binary",
            tie_csv.to_str().unwrap(),
            "--y",
            "y",
            "--p",
            "base_p",
            "--max-group-tpr-gap",
            "0.1",
        ])
        .assert()
        .code(2);

    // Duplicate CSV header rejected
    let dup_csv = dir.path().join("dup.csv");
    fs::write(&dup_csv, "y,p,p\n1,0.9,0.1\n0,0.9,0.1\n").unwrap();
    Command::cargo_bin("evalkeel")
        .unwrap()
        .args(["binary", dup_csv.to_str().unwrap(), "--y", "y", "--p", "p"])
        .assert()
        .code(2);

    // Extreme values with --cap 1 do NOT panic
    let huge_csv = dir.path().join("huge.csv");
    fs::write(&huge_csv, "actual,p50\n1e25,1e25\n1e25,1e25\n").unwrap();
    Command::cargo_bin("evalkeel")
        .unwrap()
        .args([
            "regress",
            huge_csv.to_str().unwrap(),
            "--y",
            "actual",
            "--pred",
            "p50",
            "--cap",
            "1",
            "--format",
            "agent",
        ])
        .assert()
        .success();

    // All-zero y with non-zero pred fails --max-wmape
    let zero_csv = dir.path().join("zero.csv");
    fs::write(&zero_csv, "actual,p50\n0,500\n0,500\n").unwrap();
    Command::cargo_bin("evalkeel")
        .unwrap()
        .args([
            "regress",
            zero_csv.to_str().unwrap(),
            "--y",
            "actual",
            "--pred",
            "p50",
            "--max-wmape",
            "0.05",
            "--format",
            "agent",
        ])
        .assert()
        .code(1);
}

#[test]
fn skill_prints_and_installs() {
    let dir = tempdir().unwrap();
    Command::cargo_bin("evalkeel")
        .unwrap()
        .args([
            "skill",
            "--install",
            dir.path().to_str().unwrap(),
            "--format",
            "human",
        ])
        .assert()
        .success();
    assert!(dir.path().join("evalkeel/SKILL.md").exists());
}
