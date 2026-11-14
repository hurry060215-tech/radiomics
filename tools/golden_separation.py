"""Freeze hand separation directions and independent SciPy/HiGHS LP certificates."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linprog


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("Refusing existing output; produce a fresh separation candidate.")
    definitions = [
        ("complete", [[-2.],[-1.],[1.],[2.]], [0,0,1,1], "separated", [0.,1.], 3.),
        ("quasi", [[-1.],[0.],[0.],[1.]], [0,0,1,1], "separated", [0.,1.], 2.),
        ("overlap", [[-2.],[-1.],[1.],[2.]], [0,1,0,1], "overlap", [0.,0.], 0.),
        ("rank_deficient", [[1.,2.],[2.,4.],[3.,6.],[4.,8.]], [0,0,1,1], "rank_deficient", None, None),
        ("complete_large_scale", [[-2e12],[-1e12],[1e12],[2e12]], [0,0,1,1], "separated", [0.,1.], 3.),
        ("complete_small_scale", [[-2e-12],[-1e-12],[1e-12],[2e-12]], [0,0,1,1], "separated", [0.,1.], 3.),
    ]
    cases = []
    for name, raw, labels, expected, hand_beta, hand_objective in definitions:
        design = np.c_[np.ones(len(raw)), raw]
        scale = np.max(np.abs(design), axis=0)
        scale[scale == 0] = 1
        normalized = design / scale
        signed = normalized * (2*np.asarray(labels)-1)[:, None]
        singular_values = np.linalg.svd(normalized, compute_uv=False)
        rank = int(np.sum(singular_values > singular_values.max() * 1e-12))
        case = {"id": name, "x": raw, "y": labels, "expected_diagnostic": expected,
                "column_maxabs_scales_including_intercept": scale.tolist(), "scaled_augmented_design": normalized.tolist(),
                "singular_values": singular_values.tolist(), "scaled_augmented_rank": rank,
                "hand_certificate_scaled_beta": hand_beta, "hand_optimal_margin_sum": hand_objective}
        if rank == normalized.shape[1]:
            result = linprog(-signed.sum(axis=0), A_ub=-signed, b_ub=np.zeros(len(raw)),
                             bounds=[(-1.,1.)] * normalized.shape[1], method="highs",
                             options={"primal_feasibility_tolerance": 1e-9, "dual_feasibility_tolerance": 1e-9})
            if not result.success:
                raise RuntimeError("Independent reference LP did not reach optimality.")
            margins = signed @ result.x
            case["reference_lp"] = {"solver": "SciPy1.13.1.optimize.linprog/HiGHS", "status": int(result.status),
                                    "scaled_beta": result.x.tolist(), "signed_margins": margins.tolist(), "margin_sum": float(margins.sum())}
            assert abs(margins.sum()-hand_objective) <= 1e-9
            assert margins.min() >= -1e-9
            hand_margins = signed @ hand_beta
            case["hand_signed_margins"] = hand_margins.tolist()
            assert hand_margins.min() >= 0
            assert abs(hand_margins.sum()-hand_objective) <= 1e-12
        else:
            case["reference_lp"] = None
            case["native_rule"] = "Reject rank-deficient design before solving logistic or LP."
        cases.append(case)
    output = {"schema": "python_golden_v1.logistic_separation.reference.v1", "license": "CC0-1.0 original hand data",
              "reference_packages": {"numpy":"1.23.0", "scipy":"1.13.1"},
              "specification": "Augment intercept, divide each design column by maxabs (zero column scale1), rank-check scaled design; signed rows=(2y-1)*scaled_design. beta bounds[-1,1], all signed margins>=0, maximize their sum. Positive objective proves a nonzero separation direction.",
              "tolerances": {"svd_relative_rank_threshold":1e-12, "certificate_bound_atol":1e-9,
                             "certificate_row_feasibility": "z_i >= -1e-9*(1+sum(abs(scaled_design_row_i)))",
                             "objective_sign_tolerance": "n_rows*(n_features+1)*1e-9",
                             "ambiguous_positive_objective": "0<objective<=sign_tolerance must be indeterminate/error, never used to assert a finite MLE."},
              "native_three_state_policy": {"separated": "Optimal margin sum>sign_tolerance and a verified certificate with all recomputed margins>=0 and valid coefficient bounds.",
                                            "indeterminate": "Any recomputed negative margin, even within row feasibility tolerance, or a small positive objective requires indeterminate/error unless a verified nonnegative certificate is obtained.",
                                            "no_separation_detected": "Optimal zero objective with valid zero-direction certificate means no direction detected within declared resolution, not proof of finite MLE. Clinical information matrix and convergence checks remain required."},
              "certificate_comparison": "Do not compare beta vectors: LP optima need not be unique. Independently recompute feasibility and objective from returned beta and original labels/scaled design. Complete-separation optima can have zero margins; this LP establishes separation existence, not complete-versus-quasi subtype.",
              "clinical_policy": "Separation/rank diagnostics precede unpenalized inference. Complete-case scope, observed-information invertibility and optimizer convergence must also pass. Prediction LR requires positive L2.",
              "cases": cases}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes((json.dumps(output,indent=2,allow_nan=False)+"\n").encode())
    print(args.output)


if __name__ == "__main__":
    main()
