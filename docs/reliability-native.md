# Native single-rating intraclass correlation

`reliability::icc(&NumericBuffer, IccType, confidence)` accepts a complete,
balanced subject-by-reader matrix: one subject per row, one reader per column.
At least two subjects and two readers are required. Missing values, invalid
buffer dimensions and invalid confidence are explicit errors; there is no
automatic omission or imputation. The public adapter separately validates
subject/reader identity and unique subject-reader pairs.

Choose the type explicitly:

- `IccType::Icc21`, serialized as `ICC(2,1)`: two-way random-effects absolute
  agreement, single rating.
- `IccType::Icc31`, serialized as `ICC(3,1)`: two-way fixed-reader consistency,
  single rating.

Both use the same balanced two-way ANOVA mean squares. The random-reader
estimate includes the reader mean-square adjustment; the consistency estimate
removes that component. Confidence limits use the Shrout–Fleiss ANOVA/F
construction frozen from public Pingouin 0.5.5, with presentation rounding
disabled. All F probabilities and quantiles call the shared native
`statistics::Distribution` implementation. There is no separate F CDF here.

The result reports its type, subject/reader counts, confidence, estimate and
unrounded limits, F statistic, numerator/denominator degrees of freedom,
p-value and all three ANOVA mean squares. Negative estimates and confidence
bounds are retained. No value is clipped to zero to satisfy a reliability
threshold.

Degenerate inputs retain the reference's numerical state. Completely constant
ratings produce NaN for the estimate, F, p-value and confidence limits. Perfect
agreement with varying subjects has estimate one, infinite F, p-value zero and
undefined confidence limits in the frozen reference. Results contain explicit
quality diagnostics and serialize NaN/Inf using the shared scalar/vector
serializers; undefined components are not replaced by zero or null.

The independently frozen fixture checks balanced data, reader bias, low
agreement with negative estimates, perfect agreement, constant ratings and
missing-value failure for both types. Estimates, F, p-values and unrounded
confidence limits pass `atol=1e-10, rtol=1e-8`; nonfinite states match by position.
These results do not implement average-measure ICC, missing/unbalanced mixed
models, generalized reader designs or an unspecified ICC threshold interface.

Definition and reference provenance:
[Pingouin 0.5.5 public reliability source](https://github.com/raphaelvallat/pingouin/blob/v0.5.5/src/pingouin/reliability.py),
`golden/python_golden_v1/icc_reference.json` and its pinned dependency lock.
