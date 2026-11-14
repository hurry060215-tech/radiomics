# Native observed-data baseline tables

`evaluation::baseline::baseline_table` accepts a `TabularData`, an explicit
ordered class-name vector, one optional class index per row, selected feature
names, optional cohort labels and a `BaselineConfig`.

Unlike complete-case Logistic inference, a baseline table rejects **any missing
outcome**. It never silently deletes those rows. Within each explicit cohort,
each variable is summarized using its own observed values within each outcome
class. Missingness in another variable does not remove a row from this summary.
There is no imputation, encoding, feature selection or normalization.

Numeric summaries retain observed count and missing count, mean, sample SD
(`ddof=1`), median and linearly interpolated q25/q75. An empty observed group
retains NaN summary values; a single observation has a defined mean/quantiles
but undefined sample SD. Categorical summaries retain counts and proportions
among the **nonmissing observations for that variable and outcome class**.
Missing values have a separate count and are not invented as a category or
included in the proportion denominator. Empty categorical groups have empty
count/proportion maps.

Input features are aligned by name with their declared numeric/categorical
types. Feature order and declared outcome-class order are retained. Cohorts are
reported separately in deterministic label order; data from different cohorts
are never pooled to obtain an association test for an individual cohort.

`BaselineConfig` has explicit continuous and categorical tests:

- Continuous `Auto` chooses two-sided Mann–Whitney for two outcome classes and
  tie-corrected Kruskal–Wallis for multiclass outcomes. Explicit Student t,
  Welch, MW or KW choices are also available with task-compatible groups.
- Categorical `Auto` uses two-sided Fisher only for a 2×2 table with any expected
  count below five. All other applicable tables use Pearson chi-square without
  Yates correction. In particular, a sparse 2×3 table is still chi-square; it
  is not silently treated as a multiclass Fisher test.
- Explicit Fisher on a non-2×2 table is an error. A constant categorical
  variable or a variable with an empty observed outcome group retains its
  summaries, undefined statistic/p-value and an explanatory diagnostic.

Every numerical test, mean/variance and quantile calls the existing shared
`statistics` module. There is no second test or CDF engine. Student/Welch need
at least two observed values per outcome group; inadequate groups retain their
summaries with an insufficient-observations diagnostic. Undefined t-test
statistics retain the shared test's explicit nonfinite state. Direct undefined
KW requests keep the shared function's explicit error behavior.

The typed result includes the input row count, class/cohort/feature order,
configuration, per-variable group summaries, resolved test name, statistic and
p-value, categorical level order, raw contingency counts and expected counts.
Nonfinite summary/test values serialize through the shared NaN/Inf convention.
Rows are retained even when a variable is entirely missing; zero feature values
are never fabricated to replace missing observations.

The numerical kernel identifies its analysis unit as supplied rows. Public
patient-ID and analysis-unit safeguards remain the responsibility of the
clinical adapter. The frozen binary fixture contains 95 labeled **sample
rows**, assembled after explicitly excluding one source row with a missing
outcome; the multiclass fixture contains 96 sample rows. Both contain repeated
lesions per patient and are mathematical references rather than evidence of
95/96 independent patients.

Tests compare every group count, missing count, sample SD, quantile, categorical
denominator, test choice/statistic/p-value, contingency and expected count in
the independently frozen seven-variable binary/multiclass fixtures. Additional
tests cover missing-outcome rejection, forced non-2×2 Fisher rejection, named
column reordering, JSON preservation, explicit frozen Student/Welch results and
cohort separation with undefined one-observation SD.
