use super::*;
pub(super) fn header(path: &Path) -> RadResult<Vec<String>> {
    let mut reader = csv::ReaderBuilder::new()
        .from_path(path)
        .map_err(csv_error)?;
    let headers = reader
        .byte_headers()
        .map_err(csv_error)?
        .iter()
        .map(|v| {
            std::str::from_utf8(v)
                .map(str::to_string)
                .map_err(|_| schema("CSV headers must be valid UTF8."))
        })
        .collect::<RadResult<Vec<_>>>()?;
    unique_nonempty(&headers, "CSV column names")?;
    Ok(headers)
}
pub(super) fn read(path: &Path, role: &str, projection: &Projection) -> RadResult<RawTable> {
    let mut reader = csv::ReaderBuilder::new()
        .from_path(path)
        .map_err(csv_error)?;
    let headers = reader
        .byte_headers()
        .map_err(csv_error)?
        .iter()
        .map(|v| {
            std::str::from_utf8(v)
                .map(str::to_string)
                .map_err(|_| schema("CSV headers must be valid UTF8."))
        })
        .collect::<RadResult<Vec<_>>>()?;
    let mut table = raw_shell(
        path,
        role,
        headers.clone(),
        projection.numeric.clone(),
        projection,
        0,
    )?;
    let numeric: Vec<_> = projection
        .numeric
        .iter()
        .enumerate()
        .filter_map(|(j, name)| {
            headers
                .iter()
                .position(|h| h == name)
                .map(|source| (source, j, name))
        })
        .collect();
    let text: Vec<_> = projection
        .text
        .keys()
        .filter_map(|name| {
            headers
                .iter()
                .position(|h| h == name)
                .map(|source| (source, name))
        })
        .collect();
    for (_, _, name) in &numeric {
        table
            .report
            .source_dtypes
            .insert((*name).clone(), "csv_utf8".into());
        table.report.conversions.push(ColumnConversion {
            column: (*name).clone(),
            source_dtype: "csv_utf8".into(),
            conversion: "parse_f64_with_declared_missing_policy".into(),
        });
    }
    for (_, name) in &text {
        table
            .report
            .source_dtypes
            .insert((*name).clone(), "csv_utf8".into());
    }
    let missing: Vec<_> = projection
        .missing_values
        .iter()
        .map(|v| v.as_bytes())
        .collect();
    let mut record = csv::ByteRecord::new();
    let mut rows = 0usize;
    while reader.read_byte_record(&mut record).map_err(csv_error)? {
        let start = table.numeric.values.len();
        table
            .numeric
            .values
            .try_reserve(table.numeric.cols)
            .map_err(|_| invalid("CSV numeric allocation failed."))?;
        table
            .numeric
            .missing
            .try_reserve(table.numeric.cols)
            .map_err(|_| invalid("CSV missing-state allocation failed."))?;
        table.numeric.values.resize(start + table.numeric.cols, 0.0);
        table
            .numeric
            .missing
            .resize(start + table.numeric.cols, true);
        for &(source, destination, name) in &numeric {
            let field = &record[source];
            if missing.contains(&field) {
                continue;
            }
            let text = std::str::from_utf8(field).map_err(|_| {
                schema(format!(
                    "CSV numeric '{name}' is not UTF8 on row {}.",
                    rows + 2
                ))
            })?;
            let value = text.trim().parse::<f64>().map_err(|_| {
                schema(format!(
                    "CSV numeric '{name}' cannot parse '{text}' on row {}.",
                    rows + 2
                ))
            })?;
            if let Some(value) = numeric_value(value, projection.nan_as_missing, name)? {
                table.numeric.values[start + destination] = value;
                table.numeric.missing[start + destination] = false;
            }
        }
        for &(source, name) in &text {
            let field = &record[source];
            let value = if missing.contains(&field) {
                None
            } else {
                Some(
                    std::str::from_utf8(field)
                        .map_err(|_| schema(format!("CSV string '{name}' is not UTF8.")))?
                        .to_string(),
                )
            };
            table.text.get_mut(name).unwrap().push(value);
        }
        rows = rows
            .checked_add(1)
            .ok_or_else(|| invalid("CSV row count overflows usize."))?;
    }
    table.numeric.rows = rows;
    table.report.rows = rows;
    Ok(table)
}
