use ov_radiomics_core::statistics::*;
use serde_json::Value;
fn reference() -> Value {
    serde_json::from_str(include_str!(
        "../../../golden/python_golden_v1/statistics_reference.json"
    ))
    .unwrap()
}
fn scalar(value: &Value) -> f64 {
    match value.as_str() {
        Some("NaN") => f64::NAN,
        Some("Inf") => f64::INFINITY,
        Some("-Inf") => f64::NEG_INFINITY,
        _ => value.as_f64().unwrap(),
    }
}
fn close(actual: f64, expected: &Value) {
    let expected = scalar(expected);
    if expected.is_nan() {
        assert!(actual.is_nan());
    } else if expected.is_infinite() {
        assert_eq!(actual, expected);
    } else {
        assert!(
            (actual - expected).abs() <= 1e-10 + 1e-8 * expected.abs(),
            "{actual:.17e} != {expected:.17e}"
        );
    }
}
fn values(value: &Value) -> Vec<f64> {
    value.as_array().unwrap().iter().map(scalar).collect()
}
#[test]
fn native_distribution_tails_and_quantiles_match_frozen_scipy() {
    for case in reference()["distributions"].as_array().unwrap() {
        let p = &case["parameters"];
        let d = match case["distribution"].as_str().unwrap() {
            "normal" => Distribution::Normal,
            "student_t" => Distribution::StudentT {
                df: p["df"].as_f64().unwrap(),
            },
            "chi_square" => Distribution::ChiSquare {
                df: p["df"].as_f64().unwrap(),
            },
            "f" => Distribution::F {
                dfn: p["dfn"].as_f64().unwrap(),
                dfd: p["dfd"].as_f64().unwrap(),
            },
            _ => unreachable!(),
        };
        for v in case["evaluations"].as_array().unwrap() {
            let x = scalar(&v["x"]);
            close(d.cdf(x).unwrap(), &v["cdf"]);
            close(d.sf(x).unwrap(), &v["sf"]);
        }
        for q in case["quantiles"].as_array().unwrap() {
            let p = scalar(&q["p"]);
            let actual = d.quantile(p).unwrap();
            if scalar(&q["value"]).is_finite() {
                assert!(
                    (actual - scalar(&q["value"])).abs()
                        <= 1e-10 + 1e-8 * scalar(&q["value"]).abs(),
                    "{d:?} p={p}: {actual:.17e} != {:?}",
                    q["value"]
                );
            } else {
                close(actual, &q["value"]);
            }
        }
    }
}
#[test]
fn native_hypothesis_tests_match_frozen_scipy_and_nonfinite_states() {
    let reference = reference();
    let r = &reference["hypothesis_tests"];
    for (name, equal) in [("student_t", true), ("welch", false)] {
        let v = &r[name];
        let result = t_test(&values(&v["a"]), &values(&v["b"]), equal).unwrap();
        close(result.statistic, &v["statistic"]);
        close(result.p_value, &v["p_value"]);
        close(result.degrees_of_freedom, &v["df"]);
    }
    for name in ["chi_square_2x2", "chi_square_2x3"] {
        let v = &r[name];
        let table: Vec<Vec<u64>> = serde_json::from_value(v["table"].clone()).unwrap();
        let result = chi_square(&table).unwrap();
        close(result.statistic, &v["statistic"]);
        close(result.p_value, &v["p_value"]);
        assert_eq!(
            result.degrees_of_freedom,
            v["df"].as_u64().unwrap() as usize
        );
        for (actual, expected) in result.expected.values.iter().zip(
            v["expected_counts"]
                .as_array()
                .unwrap()
                .iter()
                .flat_map(|r| r.as_array().unwrap()),
        ) {
            close(*actual, expected);
        }
    }
    for v in r["fisher_2x2"].as_array().unwrap() {
        let table: [[u64; 2]; 2] = serde_json::from_value(v["table"].clone()).unwrap();
        let result = fisher_exact(table).unwrap();
        close(result.odds_ratio, &v["odds_ratio"]);
        close(result.p_value, &v["p_value"]);
    }
    let same = t_test(&[1.; 3], &[1.; 3], true).unwrap();
    let unequal = t_test(&[1.; 3], &[2.; 3], false).unwrap();
    close(same.statistic, &r["constant_equal_t"]["value"][0]);
    close(same.p_value, &r["constant_equal_t"]["value"][1]);
    close(
        unequal.statistic,
        &r["constant_different_welch"]["value"][0],
    );
    close(unequal.p_value, &r["constant_different_welch"]["value"][1]);
    let json = serde_json::to_string(&same).unwrap();
    assert!(json.contains("NaN"));
    let restored: TTestResult = serde_json::from_str(&json).unwrap();
    assert!(restored.statistic.is_nan());
    assert!(chi_square(&[vec![0, 0], vec![0, 1]]).is_err());
}
