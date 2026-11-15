# Research combinations: pre-implementation Golden freeze candidate

This freeze specifies keyed feature concatenation, paired Delta and basic
Lloyd KMeans before their native implementation. The main worker independently
rebuilt and structurally verified both the original freeze and supplementary
direct-distance freeze before approving native execution. The reference generator uses public NumPy 1.23.0 and
scikit-learn 1.0.2; no native output enters the expected values.

## Scope and contracts

`research_spec.py` is a small auditable independent specification for table
transformations. Concatenation requires exact explicit metadata keys, unique
source prefixes, equal key populations and caller-declared completed registration.
It preserves anchor row order, source/feature order, source sample IDs and source
parameters. Only explicitly declared modality/region differences may vary;
missing metadata does not silently equal a known value. IDs and outcomes remain
outside the numeric matrix.

Delta uses patient/lesion/modality/region keys and exactly the selected two
timepoints. It produces difference, ratio and relative change in explicit input
feature order. Missing propagation precedes denominator checks. Zero denominators
default to error; explicit missing policy preserves an observed difference while
marking ratios missing. Extra timepoints, unpaired/duplicate keys, metadata
conflicts and arithmetic overflow are errors.

The habitat fixture fits Standard scaling using only eight training observations,
transforms explicit raw-unit initial centers using that fitted state, and runs
Lloyd with `n_init=1`, sklearn 1.0.2's `algorithm="full"`, `tol=1e-12` and an
explicit iteration limit. It freezes scaler, transformed data, centers, labels,
inertia and evaluation distances. A separate one-iteration fixture returns a
non-fixed-point sklearn result: native implementation must report nonconvergence
and refuse to treat it as a successful model. Empty clusters are explicit native
errors, without sklearn's hidden relocation behavior.

## Approved midpoint correction before native implementation

The old candidate's third evaluation point `[4.5,4.5]` was incorrectly described
by its Python test as an exact assignment tie. Fitted sklearn centers differ at
floating precision, so its distances were `1.403292830891247` and
`1.4032928308912467` and its actual frozen label was already `1`.

The main worker explicitly approved changing this primary evaluation point to
`[4.6,4.5]` before native implementation. The revised reference changes exactly
two distance scalars to `1.420942265919669` and `1.3858626860607024`, updates the
input hash, and adds two records. All training state, labels, centers, inertia,
concatenation, Delta, failure expectations and tolerances remain unchanged.

The old point and exact old sklearn result remain in
`habitat_old_midpoint_diagnostic`, explicitly outside exact tie acceptance.
`habitat_exact_tie` independently hand-calculates scaled centers `[[-1,0],[1,0]]`
and query `[0,0]`, squared distances `[1,1]`, and lowest-index label `0`.
No tolerance has been widened.

An additional pre-native-test distance definition is frozen separately:
`habitat.evaluation_distances_euclidean_direct` is NumPy's direct Euclidean norm
of feature differences, using only the public reference training scaler and
frozen centers. It uses the existing deterministic scalar tolerance
`atol=1e-10, rtol=1e-8`. The original sklearn `evaluation_distances` values are
unchanged and retained as an expanded-norm diagnostic: cancellation near a
center can produce a spurious distance around 1e-8. That diagnostic is not the
native mathematical distance definition or an acceptance gate. This addition
was explicitly required by the main worker before native execution/testing;
no native output was consulted.

| Input / reference | SHA256 |
|---|---|
| research_spec.py | b3abdee7da1f6dda8a0b6c5019b2192209d31638c9d8e2641396b5673c04ca68 |
| revised research_inputs.json | 0e2e4790533bd0f23711a879c5d0af0b2cff3f07b425fd43f9ffd153e8a17a70 |
| revised research_reference.json | 3cb7c29e28ef47aa0205cbf6656884f1092932f7312762731a114853f7fb03a8 |

Rebuild into a new candidate path, never over the frozen fixture:

```powershell
python tools/golden_research.py --output artifacts/research-candidate.json
python -m pytest tests/test_golden_research.py -q
```

These are correctness references, not performance measurements or biological
tissue labels. Per-patient cluster numbers have no automatic cross-patient
histologic meaning.
