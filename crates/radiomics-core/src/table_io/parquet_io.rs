use super::*;
use parquet::basic::{ConvertedType, LogicalType, Type as PhysicalType};
use parquet::column::reader::{ColumnReader, ColumnReaderImpl};
use parquet::data_type::{
    BoolType, ByteArrayType, DataType, DoubleType, FloatType, Int32Type, Int64Type,
};
use parquet::file::reader::{FileReader, SerializedFileReader};

fn names(reader: &SerializedFileReader<File>) -> RadResult<Vec<String>> {
    let fields = reader.metadata().file_metadata().schema().get_fields();
    if fields.iter().any(|f| !f.is_primitive())
        || reader
            .metadata()
            .file_metadata()
            .schema_descr()
            .columns()
            .iter()
            .any(|d| d.max_rep_level() != 0 || d.max_def_level() > 1)
    {
        return Err(RadiomicsError::new(ErrorCode::Unsupported,"Only flat required/optional scalar Parquet columns are supported; nested/list/repeated columns are not inferred."));
    }
    let names: Vec<_> = fields.iter().map(|f| f.name().to_string()).collect();
    unique_nonempty(&names, "Parquet column names")?;
    Ok(names)
}
pub(super) fn header(path: &Path) -> RadResult<Vec<String>> {
    names(&SerializedFileReader::new(File::open(path)?).map_err(parquet_error)?)
}
fn batches<T: DataType>(
    mut reader: ColumnReaderImpl<T>,
    optional: bool,
    rows: usize,
    mut accept: impl FnMut(usize, Option<&T::T>) -> RadResult<()>,
) -> RadResult<()> {
    let mut offset = 0usize;
    let mut values = Vec::new();
    let mut definitions = Vec::new();
    while offset < rows {
        values.clear();
        definitions.clear();
        let (records, nonmissing, levels) = reader
            .read_records(
                (rows - offset).min(1024),
                optional.then_some(&mut definitions),
                None,
                &mut values,
            )
            .map_err(parquet_error)?;
        if records == 0
            || records > rows - offset
            || levels != records
            || values.len() != nonmissing
            || (optional && definitions.len() != records)
        {
            return Err(schema(
                "Parquet column length/definition levels disagree with declared row-group records.",
            ));
        }
        let mut cursor = 0;
        for (i, level) in definitions
            .iter()
            .copied()
            .chain(std::iter::repeat(1))
            .take(records)
            .enumerate()
        {
            let present = if optional {
                match level {
                    0 => false,
                    1 => true,
                    _ => return Err(schema("Unexpected flat Parquet definition level.")),
                }
            } else {
                true
            };
            let value = if present {
                let value = values
                    .get(cursor)
                    .ok_or_else(|| schema("Parquet nonnull value count is too short."))?;
                cursor += 1;
                Some(value)
            } else {
                None
            };
            accept(offset + i, value)?;
        }
        if cursor != values.len() {
            return Err(schema(
                "Parquet nonnull values and definition levels disagree.",
            ));
        }
        offset += records;
    }
    Ok(())
}
fn integer_type(
    logical: &Option<LogicalType>,
    converted: ConvertedType,
    physical_bits: u8,
) -> RadResult<(u8, bool)> {
    let (width,signed)=match logical {
        Some(LogicalType::Integer{bit_width,is_signed})=>(*bit_width as u8,*is_signed),
        None=>match converted {ConvertedType::UINT_8=>(8,false),ConvertedType::UINT_16=>(16,false),ConvertedType::UINT_32=>(32,false),ConvertedType::UINT_64=>(64,false),ConvertedType::INT_8=>(8,true),ConvertedType::INT_16=>(16,true),ConvertedType::INT_32=>(32,true),ConvertedType::INT_64=>(64,true),ConvertedType::NONE=>(physical_bits,true),_=>return Err(RadiomicsError::new(ErrorCode::Unsupported,"Date/decimal/time and other logical integer annotations are not ordinary numeric/categorical columns."))},
        _=>return Err(RadiomicsError::new(ErrorCode::Unsupported,"Unsupported Parquet logical integer annotation.")),
    };
    if !matches!(width, 8 | 16 | 32 | 64) || width > physical_bits {
        return Err(schema("Invalid Parquet integer width annotation."));
    }
    Ok((width, signed))
}
fn integer(
    value: i64,
    physical_bits: u8,
    width: u8,
    signed: bool,
    numeric: bool,
) -> RadResult<(Option<f64>, Option<String>)> {
    if signed {
        if width < 64 && (value < -(1i64 << (width - 1)) || value > (1i64 << (width - 1)) - 1) {
            return Err(schema("Parquet integer exceeds its signed logical width."));
        }
        if numeric {
            let converted = value as f64;
            if converted as i128 != i128::from(value) {
                return Err(schema(
                    "Int64 feature cannot be converted to f64 without real integer precision loss.",
                ));
            }
            Ok((Some(converted), None))
        } else {
            Ok((None, Some(value.to_string())))
        }
    } else {
        let value = if physical_bits == 32 {
            u64::from(value as i32 as u32)
        } else {
            value as u64
        };
        if width < 64 && value > (1u64 << width) - 1 {
            return Err(schema(
                "Parquet integer exceeds its unsigned logical width.",
            ));
        }
        if numeric {
            let converted = value as f64;
            if converted as u128 != u128::from(value) {
                return Err(schema("UInt64 feature cannot be converted to f64 without real integer precision loss."));
            }
            Ok((Some(converted), None))
        } else {
            Ok((None, Some(value.to_string())))
        }
    }
}
fn set_number(
    table: &mut RawTable,
    row: usize,
    col: usize,
    value: Option<f64>,
    name: &str,
    nan_as_missing: bool,
) -> RadResult<()> {
    if let Some(value) = value {
        if let Some(value) = numeric_value(value, nan_as_missing, name)? {
            let index = row * table.numeric.cols + col;
            table.numeric.values[index] = value;
            table.numeric.missing[index] = false;
        }
    }
    Ok(())
}
pub(super) fn read(path: &Path, role: &str, projection: &Projection) -> RadResult<RawTable> {
    let reader = SerializedFileReader::new(File::open(path)?).map_err(parquet_error)?;
    let headers = names(&reader)?;
    let rows = usize::try_from(reader.metadata().file_metadata().num_rows())
        .map_err(|_| schema("Parquet row count does not fit usize."))?;
    let mut table = raw_shell(
        path,
        role,
        headers.clone(),
        projection.numeric.clone(),
        projection,
        rows,
    )?;
    if let Some(metadata) = reader.metadata().file_metadata().key_value_metadata() {
        let manifests: Vec<_> = metadata
            .iter()
            .filter(|v| v.key == FEATURE_MANIFEST_KEY)
            .collect();
        if manifests.len() > 1 {
            return Err(schema("Parquet contains duplicate feature manifest keys."));
        }
        if let Some(entry) = manifests.first() {
            let manifest: FeatureTableManifest = serde_json::from_str(
                entry
                    .value
                    .as_deref()
                    .ok_or_else(|| schema("Parquet feature manifest has no value."))?,
            )?;
            manifest.validate()?;
            if manifest.table_sha256.is_some() {
                return Err(schema("Embedded feature manifest must not contain the same file's SHA256 (circular hash)."));
            }
            table.embedded_manifest = Some(manifest);
        }
    }
    for (source, name) in headers.iter().enumerate() {
        let numeric = projection.numeric.iter().position(|n| n == name);
        let text = projection.text.get(name).copied();
        if numeric.is_none() && text.is_none() {
            continue;
        }
        let descriptor = reader
            .metadata()
            .file_metadata()
            .schema_descr()
            .column(source);
        let logical = descriptor.logical_type_ref().cloned();
        let converted = descriptor.converted_type();
        let dtype = format!("{:?}/{logical:?}", descriptor.physical_type());
        table
            .report
            .source_dtypes
            .insert(name.clone(), dtype.clone());
        let null_only = logical == Some(LogicalType::Unknown);
        let utf8 = logical == Some(LogicalType::String) || converted == ConvertedType::UTF8;
        let categorical = text == Some(TextRole::Categorical);
        let physical = descriptor.physical_type();
        if null_only {
            if text == Some(TextRole::Id) {
                return Err(schema(format!(
                    "Parquet ID '{name}' must be UTF8 even when all values are null."
                )));
            }
            if !matches!(physical, PhysicalType::INT32 | PhysicalType::INT64) {
                return Err(RadiomicsError::new(
                    ErrorCode::Unsupported,
                    "Logical NULL columns currently require an integer physical representation.",
                ));
            }
        } else if numeric.is_some() {
            match physical {
                PhysicalType::INT32 => {
                    integer_type(&logical, converted, 32)?;
                }
                PhysicalType::INT64 => {
                    integer_type(&logical, converted, 64)?;
                }
                PhysicalType::FLOAT | PhysicalType::DOUBLE if logical.is_none() => (),
                _ => {
                    return Err(schema(format!(
                        "Parquet numeric '{name}' has an unsupported physical/logical type."
                    )))
                }
            }
        } else if utf8 && physical == PhysicalType::BYTE_ARRAY {
            // Explicit UTF8 values, including empty strings, remain strings.
        } else if categorical {
            match physical {PhysicalType::INT32=>{integer_type(&logical,converted,32)?;},PhysicalType::INT64=>{integer_type(&logical,converted,64)?;},PhysicalType::BOOLEAN if logical.is_none()=>(),_=>return Err(schema(format!("Parquet categorical '{name}' requires UTF8 or integer/bool; float categories require explicit string encoding.")))}
        } else {
            return Err(schema(format!(
                "Parquet ID/string metadata '{name}' must be UTF8."
            )));
        }
        let optional = descriptor.max_def_level() == 1;
        let conversion = if null_only {
            "logical_null_to_explicit_missing"
        } else if numeric.is_some() {
            "checked_numeric_to_f64"
        } else if categorical && !utf8 {
            "integer_or_bool_to_canonical_categorical_string"
        } else {
            "utf8_identity"
        };
        table.report.conversions.push(ColumnConversion {
            column: name.clone(),
            source_dtype: dtype,
            conversion: conversion.into(),
        });
        let mut base_row = 0usize;
        for group_index in 0..reader.num_row_groups() {
            let group = reader.get_row_group(group_index).map_err(parquet_error)?;
            let count = usize::try_from(group.metadata().num_rows())
                .map_err(|_| schema("Parquet row-group row count is invalid."))?;
            if base_row.checked_add(count).is_none_or(|n| n > rows) {
                return Err(schema("Parquet row groups exceed file row count."));
            }
            let column = group.get_column_reader(source).map_err(parquet_error)?;
            macro_rules! integer_column {
                ($reader:expr,$type:ty,$bits:expr)=>{{
                    let annotation=if null_only {($bits,true)} else {integer_type(&logical,converted,$bits)?};
                    if !null_only && numeric.is_none() && !categorical {return Err(schema(format!("Parquet ID/string metadata '{name}' must be UTF8, not a numeric column.")));}
                    batches::<$type>($reader,optional,count,|row,value| {if null_only {if value.is_some() {return Err(schema("Parquet logical NULL column contains a nonnull value."));}return Ok(());}if let Some(&value)=value {let (number,string)=integer(i64::from(value),$bits,annotation.0,annotation.1,numeric.is_some())?;if let Some(j)=numeric {set_number(&mut table,base_row+row,j,number,name,projection.nan_as_missing)?;} else {table.text.get_mut(name).unwrap()[base_row+row]=string;}}Ok(())})?;
                }};
            }
            match column {
                ColumnReader::Int32ColumnReader(reader) => integer_column!(reader, Int32Type, 32),
                ColumnReader::Int64ColumnReader(reader) => integer_column!(reader, Int64Type, 64),
                ColumnReader::DoubleColumnReader(reader) => {
                    if numeric.is_none() || logical.is_some() {
                        return Err(schema(format!("Parquet float column '{name}' must have an ordinary numeric/regression role; float categorical values require explicit string encoding.")));
                    }
                    let j = numeric.unwrap();
                    batches::<DoubleType>(reader, optional, count, |row, v| {
                        set_number(
                            &mut table,
                            base_row + row,
                            j,
                            v.copied(),
                            name,
                            projection.nan_as_missing,
                        )
                    })?;
                }
                ColumnReader::FloatColumnReader(reader) => {
                    if numeric.is_none() || logical.is_some() {
                        return Err(schema(format!("Parquet float column '{name}' must have an ordinary numeric/regression role; float categorical values require explicit string encoding.")));
                    }
                    let j = numeric.unwrap();
                    batches::<FloatType>(reader, optional, count, |row, v| {
                        set_number(
                            &mut table,
                            base_row + row,
                            j,
                            v.map(|&v| f64::from(v)),
                            name,
                            projection.nan_as_missing,
                        )
                    })?;
                }
                ColumnReader::ByteArrayColumnReader(reader) => {
                    if numeric.is_some() || !utf8 {
                        return Err(schema(format!("Parquet string role '{name}' requires UTF8 logical text; numeric strings are not silently parsed as typed Parquet numbers.")));
                    }
                    batches::<ByteArrayType>(reader, optional, count, |row, value| {
                        table.text.get_mut(name).unwrap()[base_row + row] = value
                            .map(|v| v.as_utf8().map(str::to_string).map_err(parquet_error))
                            .transpose()?;
                        Ok(())
                    })?;
                }
                ColumnReader::BoolColumnReader(reader) => {
                    if !categorical || numeric.is_some() || logical.is_some() {
                        return Err(schema(format!("Parquet bool '{name}' is supported only for an explicit categorical/outcome role, not IDs or numeric features.")));
                    }
                    batches::<BoolType>(reader, optional, count, |row, value| {
                        table.text.get_mut(name).unwrap()[base_row + row] =
                            value.map(|&v| if v { "1".into() } else { "0".into() });
                        Ok(())
                    })?;
                }
                _ => {
                    return Err(RadiomicsError::new(
                        ErrorCode::Unsupported,
                        format!("Unsupported Parquet column type for '{name}'."),
                    ))
                }
            }
            base_row += count;
        }
        if base_row != rows {
            return Err(schema(
                "Parquet row-group totals differ from file row count.",
            ));
        }
    }
    Ok(table)
}
