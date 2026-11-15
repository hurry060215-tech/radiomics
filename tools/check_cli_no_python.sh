#!/bin/sh
# Run inside the CI minimal Ubuntu image, with /src, /cli and a new /out mounted.
set -eu
if command -v python >/dev/null 2>&1 || command -v python3 >/dev/null 2>&1; then
    printf '%s\n' 'This deployment check requires an image without Python.' >&2
    exit 1
fi
cd /out
printf '%s\n' 'sample_id,patient_id,image,mask' \
    'phantom,phantom,/src/golden/ibsi_digital_v1/image.nii.gz,/src/golden/ibsi_digital_v1/mask.nii.gz' > cohort.csv
printf '%s\n' '{"extraction":{"profile":"python_golden_v1","discretization":{"method":"fixed_width","width":1},"aggregation":"direction_mean","gldm_alpha":0,"voxel_array_shift":0},"labels":[1],"preparation":[]}' > extract.json
/cli/radiomics check --cohort cohort.csv --output image-check.json
/cli/radiomics extract --cohort cohort.csv --config extract.json --output extraction
test -s extraction/features.csv
IFS= read -r header < extraction/features.csv
case "$header" in
    *original_firstorder_Mean*) ;;
    *) printf '%s\n' 'Native extraction did not export the declared first-order feature.' >&2; exit 1 ;;
esac
printf '%s\n' 'sample_id,patient_id,cohort,outcome,signal,noise' > study.csv
i=0
while test "$i" -lt 24; do
    cohort=train
    test "$i" -lt 16 || cohort=test
    test "$i" -lt 20 || cohort=external_B
    target=$((i % 2))
    signal=$((3 * target + i % 3))
    noise=$((i % 7))
    test "$i" -lt 20 || target=''
    printf 'S%s,P%s,%s,%s,%s,%s\n' "$i" "$i" "$cohort" "$target" "$signal" "$noise" >> study.csv
    i=$((i + 1))
done
printf '%s\n' '{"schema_version":1,"table":"study.csv","outcome_column":"outcome","numeric_features":["signal","noise"],"optional_metadata":{}}' > study.json
printf '%s\n' '{"schema_version":1,"table":"study.csv","numeric_features":["signal","noise"],"optional_metadata":{}}' > unlabeled-study.json
printf '%s\n' '{"task":"binary","classes":["0","1"],"fit_unit":"sample","scoring_unit":"patient","threshold":0.5,"selection_tie_atol":1e-12,"models":[{"name":"LR","candidates":[{"preprocessing":{"scaling":"standard","all_missing":"error"},"selector":null,"model":{"model":"LR","lambda":0.1,"tolerance":1e-10,"max_iterations":5000}}]}]}' > train.json
/cli/radiomics train --data study.json --config train.json --outer-cv 2 --inner-cv 2 --output model
/cli/radiomics predict --model model/model-000.rad --data unlabeled-study.json --output prediction
/cli/radiomics evaluate --predictions prediction/predictions.json --data study.json --threshold 0.5 --n-bootstrap 100 --output evaluation
test -s model/model-000.rad
test -s prediction/predictions.json
test -s evaluation/summary.json
printf '%s\n' 'Python-free NIfTI extraction, grouped training, .rad reload, unlabeled prediction and evaluation passed.'
