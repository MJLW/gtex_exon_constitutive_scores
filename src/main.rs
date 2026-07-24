use clap::Parser;
use core::f32;
use std::{fs::File, path::PathBuf};

use arrow_array::{Array, ArrowNativeTypeOp, Float32Array, RecordBatch, StringArray};
use arrow_select::concat::concat_batches;
use csv::WriterBuilder;
use parquet::arrow::arrow_reader::{ParquetRecordBatchReader, ParquetRecordBatchReaderBuilder};

mod metadata;
use crate::metadata::{DonorMetadataTable, ExonMetadataTable};

#[derive(Debug, Parser)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(long)]
    exon_counts: PathBuf,

    #[arg(long)]
    exon_metadata: PathBuf,

    #[arg(long)]
    donor_metadata: PathBuf,

    #[arg(long)]
    output: PathBuf,
}

struct Score {
    key: String,
    constitutive_score: f32,
    constitutive_tissue_variance: f32,
    constitutive_donor_variance: f32,
    n_expressed_tissues: usize,
    total_mean: f32,
    tissue_mean: f32,
    donor_mean: f32,
    total_variance: f32,
    tissue_variance: f32,
    donor_variance: f32,
}

fn find_gene_boundaries(arr: &[&str]) -> Vec<usize> {
    let len = arr.len();
    let mut result: Vec<usize> = Vec::new();

    if len == 0 {
        return result;
    }

    result.push(0);
    let mut prev = arr[0];

    for i in 1..len {
        let current = arr[i];

        if current != prev {
            result.push(i);
            prev = current;
        }
    }

    result
}

fn prepend_previous_batch(
    current_batch: &RecordBatch,
    previous_batch: &Option<RecordBatch>,
) -> Result<RecordBatch, Box<dyn std::error::Error>> {
    let batch = if let Some(carry_over) = previous_batch {
        concat_batches(
            &current_batch.schema(),
            &[carry_over.clone(), current_batch.clone()],
        )?
    } else {
        current_batch.clone()
    };

    Ok(batch)
}

fn get_batch_column_as_array<T: 'static>(batch: &RecordBatch, column_index: usize) -> &T {
    batch
        .column(column_index)
        .as_any()
        .downcast_ref::<T>()
        .unwrap()
}

fn calculate_constitutive_scores(
    read_counts: &[f32],
    exon_sizes: &[usize],
    gene_boundaries: &[(usize, usize)],
) -> Vec<f32> {
    gene_boundaries
        .into_iter()
        .map(|(start, end)| {
            let gene_reads = &read_counts[*start..*end];
            let lengths = &exon_sizes[*start..*end];

            // Calculate coverage (counts / region length)
            let coverages: Vec<f32> = gene_reads
                .iter()
                .zip(lengths)
                .map(|(x, &l)| (x, l))
                .map(|(x, l)| match l > 0 {
                    true => x / l as f32,
                    false => 0.0,
                })
                .collect();

            // Divide scores by max coverage to get a ratio of how often it is expressed
            // Vec<f32> does not have a iter.max() function, so fold into a pairwise max
            let max_coverage = coverages.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));

            // Play stupid games, win stupid prizes
            if max_coverage == 0.0 {
                return vec![f32::NAN; *end - *start];
            }

            return coverages
                .iter()
                .map(|c| c / max_coverage)
                .collect::<Vec<f32>>();
        })
        .flatten()
        .collect()
}

fn as_row_major_matrix(flat_data: Vec<f32>, n_rows: usize, n_cols: usize) -> Vec<Vec<f32>> {
    (0..n_rows)
        .into_iter()
        .map(|i| {
            (0..n_cols)
                .into_iter()
                .map(|j| flat_data[i + n_rows * j])
                .collect()
        })
        .collect()
}

fn calculate_mean(data: &Vec<f32>) -> f32 {
    let finite_data: Vec<&f32> = data.iter().filter(|&v| !v.is_finite()).collect();

    if finite_data.len() == 0 {
        return f32::NAN;
    }

    let len = finite_data.len();
    finite_data.into_iter().sum::<f32>() / (len as f32)
}

fn calculate_median(data: &Vec<f32>) -> f32 {
    let mut finite_data: Vec<&f32> = data.iter().filter(|&v| v.is_finite()).collect();

    if finite_data.len() == 0 {
        return f32::NAN;
    }

    finite_data.sort_unstable_by(|&a, &b| a.total_cmp(b));
    let len = finite_data.len();

    *finite_data[len / 2]
}

fn calculate_standardized_variance(data: &Vec<f32>) -> f32 {
    let len = data.len();
    let mean: f32 = data.iter().sum::<f32>() / (len as f32);

    (data.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / (len as f32))
        .div_checked(mean)
        .unwrap_or(f32::NAN)
}

fn get_sample_indices_for_tissues<'a>(
    sample_ids: &Vec<String>,
    donor_metadata: &'a DonorMetadataTable,
) -> Vec<(&'a str, Vec<usize>)> {
    donor_metadata
        .get_tissues()
        .iter()
        .map(|&t| {
            (
                t,
                donor_metadata
                    .get_tissue_samples(t)
                    .unwrap()
                    .iter()
                    .map(|ts| sample_ids.iter().position(|s| ts == s).unwrap())
                    .collect::<Vec<usize>>(),
            )
        })
        .collect()
}

fn get_sample_indices_for_donors<'a>(
    sample_ids: &Vec<String>,
    donor_metadata: &'a DonorMetadataTable,
) -> Vec<(&'a str, Vec<usize>)> {
    donor_metadata
        .get_donors()
        .iter()
        .map(|&t| {
            (
                t,
                donor_metadata
                    .get_donor_samples(t)
                    .unwrap()
                    .iter()
                    .map(|ts| sample_ids.iter().position(|s| ts == s).unwrap())
                    .collect(),
            )
        })
        .collect()
}

fn calculate_variance_over_group_medians(
    scores_row: &Vec<f32>,
    column_indices_per_group: &Vec<(&str, Vec<usize>)>,
) -> f32 {
    let medians: Vec<f32> = column_indices_per_group
        .iter()
        .map(|(_, indices)| indices.iter().map(|&i| scores_row[i]).collect())
        .map(|scores| calculate_median(&scores))
        .collect();

    calculate_standardized_variance(&medians)
}

fn calculate_mean_over_group_medians(
    scores_row: &Vec<f32>,
    column_indices_per_group: &Vec<(&str, Vec<usize>)>,
) -> f32 {
    let medians: Vec<f32> = column_indices_per_group
        .iter()
        .map(|(_, indices)| indices.iter().map(|&i| scores_row[i]).collect())
        .map(|scores| calculate_median(&scores))
        .collect();

    calculate_mean(&medians)
}

// WARN: Made to work with GTEx v7, very particular to the column order and data types.
fn read_exon_parquet(
    path: PathBuf,
    exon_metadata: &ExonMetadataTable,
    donor_metadata: &DonorMetadataTable,
) -> Result<Vec<Score>, Box<dyn std::error::Error>> {
    const CHUNK_SIZE: usize = 2048;
    let builder =
        ParquetRecordBatchReaderBuilder::try_new(File::open(path)?)?.with_batch_size(CHUNK_SIZE);

    // Parse column information from metadata
    let metadata = builder.metadata().file_metadata();
    let n_columns = metadata.schema_descr().num_columns() as usize;

    // Grab data columns, skip Description and Name columns
    let sample_ids: Vec<String> = metadata
        .schema_descr()
        .columns()
        .iter()
        .map(|c| c.name().to_string())
        .collect::<Vec<_>>()[1..n_columns - 1]
        .to_vec();

    let column_indices_per_tissue = get_sample_indices_for_tissues(&sample_ids, donor_metadata);
    let column_indices_per_donor = get_sample_indices_for_donors(&sample_ids, donor_metadata);

    let mut scores: Vec<Score> = Vec::with_capacity(metadata.num_rows() as usize);

    let mut reader: ParquetRecordBatchReader = builder.build()?;
    let mut previous_batch: Option<RecordBatch> = None;

    while let Some(current_batch) = reader.next().transpose()? {
        // Genes will overlap batch boundaries, so we need to prepend the remainder of the previous batch
        let batch = prepend_previous_batch(&current_batch, &previous_batch)?;

        let total_name_array = get_batch_column_as_array::<StringArray>(&batch, n_columns - 1);
        let ensgs: Vec<_> = total_name_array
            .iter()
            .map(|name| name.unwrap().splitn(2, '.').next().unwrap())
            .collect();

        // It processes until the last start, because that gene will likely be incomplete
        let gene_start_indices = find_gene_boundaries(&ensgs);
        let n_rows = *gene_start_indices.last().unwrap();
        let gene_boundaries: Vec<(usize, usize)> = gene_start_indices
            .windows(2)
            .map(|pair| (pair[0], pair[1]))
            .collect();

        let name_array = total_name_array.slice(0, n_rows);

        // Get exon sizes from metadata table
        let exon_sizes: Vec<usize> = name_array
            .iter()
            .take(n_rows)
            .map(|n| exon_metadata.get_exon(n.unwrap()).unwrap().size)
            .collect();

        // Process read counts
        let n_cols = n_columns - 2;
        let count_arrays: Vec<_> = (1..n_cols + 1)
            .into_iter()
            .map(|i| get_batch_column_as_array::<Float32Array>(&batch, i))
            .collect();

        // Filter to only the genes that will be used this batch
        let column_counts: Vec<Vec<f32>> = count_arrays
            .iter()
            .map(|col| {
                col.iter()
                    .take(n_rows)
                    .map(|x| x.unwrap() as f32)
                    .collect::<Vec<f32>>()
            })
            .collect();

        let sample_constitutive_scores: Vec<_> = column_counts
            .iter()
            .map(|counts| {
                calculate_constitutive_scores(
                    counts.as_slice(),
                    exon_sizes.as_slice(),
                    gene_boundaries.as_slice(),
                )
            })
            .flatten()
            .collect();

        let tissues_counts: Vec<Vec<f32>> = column_indices_per_tissue
            .iter()
            .map(|(_, indices)| {
                let tissue_donor_counts: Vec<f32> = indices
                    .into_iter()
                    .map(|&i| column_counts.get(i).unwrap().as_slice())
                    .flatten()
                    .map(|&f| f)
                    .collect();

                let row_counts = as_row_major_matrix(tissue_donor_counts, n_rows, indices.len());
                let tissue_counts: Vec<f32> = row_counts
                    .into_iter()
                    .map(|row| calculate_median(&row))
                    .collect();

                tissue_counts
            })
            .collect();

        let tissue_constitutive_scores: Vec<_> = tissues_counts
            .iter()
            .map(|tissue_counts| {
                calculate_constitutive_scores(
                    tissue_counts.as_slice(),
                    exon_sizes.as_slice(),
                    gene_boundaries.as_slice(),
                )
            })
            .flatten()
            .collect();

        // Reorder scores such that samples are stored contiguously
        let counts: Vec<f32> = column_counts.into_iter().flatten().collect();
        let row_counts: Vec<Vec<f32>> = as_row_major_matrix(counts, n_rows, n_cols);
        let row_tissue_scores: Vec<Vec<f32>> =
            as_row_major_matrix(tissue_constitutive_scores, n_rows, tissues_counts.len());

        let constitutive_donor_variances: Vec<f32> =
            as_row_major_matrix(sample_constitutive_scores, n_rows, n_cols)
                .into_iter()
                .map(|row| {
                    let tissue_variances: Vec<_> = column_indices_per_tissue
                        .iter()
                        .map(|(_, indices)| indices.iter().map(|&i| row[i]).collect::<Vec<f32>>())
                        .map(|donor_scores| calculate_standardized_variance(&donor_scores))
                        .collect();

                    calculate_standardized_variance(&tissue_variances)
                })
                .collect();

        let batch_scores: Vec<Score> = row_tissue_scores
            .iter()
            .zip(constitutive_donor_variances)
            .zip(row_counts)
            .zip(&name_array)
            .map(|(((score_row, donor_variance), counts_row), name)| Score {
                key: name.unwrap().to_string(),
                constitutive_score: calculate_median(&score_row),
                constitutive_tissue_variance: calculate_standardized_variance(&score_row),
                constitutive_donor_variance: donor_variance,
                n_expressed_tissues: score_row.iter().filter(|&&score| score > 0.0).count(),
                total_mean: calculate_mean(&counts_row),
                tissue_mean: calculate_mean_over_group_medians(
                    &counts_row,
                    &column_indices_per_tissue,
                ),
                donor_mean: calculate_mean_over_group_medians(
                    &counts_row,
                    &column_indices_per_donor,
                ),
                total_variance: calculate_standardized_variance(&counts_row),
                tissue_variance: calculate_variance_over_group_medians(
                    &counts_row,
                    &column_indices_per_tissue,
                ),
                donor_variance: calculate_variance_over_group_medians(
                    &counts_row,
                    &column_indices_per_donor,
                ),
            })
            .collect();

        scores.extend(batch_scores);

        // Keep unprocessed part of current batch to prepend to next one
        previous_batch = Some(batch.slice(n_rows, batch.num_rows() - n_rows));
    }

    // Run leftover gene from last batch
    if let Some(batch) = previous_batch {
        let name_array = get_batch_column_as_array::<StringArray>(&batch, n_columns - 1);
        let exon_sizes: Vec<usize> = name_array
            .iter()
            .map(|n| exon_metadata.get_exon(n.unwrap()).unwrap().size)
            .collect();

        let n_rows = batch.num_rows();
        let n_cols = n_columns - 2;

        let count_arrays: Vec<_> = (1..n_cols + 1)
            .into_iter()
            .map(|i| get_batch_column_as_array::<Float32Array>(&batch, i))
            .collect();

        let column_counts: Vec<Vec<f32>> = count_arrays
            .iter()
            .map(|col| col.iter().map(|x| x.unwrap() as f32).collect::<Vec<f32>>())
            .collect();

        let sample_constitutive_scores: Vec<_> = column_counts
            .iter()
            .map(|counts| {
                calculate_constitutive_scores(
                    counts.as_slice(),
                    exon_sizes.as_slice(),
                    &vec![(0, batch.num_rows())],
                )
            })
            .flatten()
            .collect();

        let tissues_counts: Vec<Vec<f32>> = column_indices_per_tissue
            .iter()
            .map(|(_, indices)| {
                let tissue_donor_counts: Vec<f32> = indices
                    .into_iter()
                    .map(|&i| column_counts.get(i).unwrap().as_slice())
                    .flatten()
                    .map(|&f| f)
                    .collect();

                let row_counts = as_row_major_matrix(tissue_donor_counts, n_rows, indices.len());
                let tissue_counts: Vec<f32> = row_counts
                    .into_iter()
                    .map(|row| calculate_median(&row))
                    .collect();

                tissue_counts
            })
            .collect();

        let constitutive_scores: Vec<_> = tissues_counts
            .iter()
            .map(|tissue_counts| {
                calculate_constitutive_scores(
                    tissue_counts.as_slice(),
                    exon_sizes.as_slice(),
                    &vec![(0, batch.num_rows())],
                )
            })
            .flatten()
            .collect();

        let counts: Vec<f32> = column_counts.into_iter().flatten().collect();
        let reordered_counts: Vec<Vec<f32>> = as_row_major_matrix(counts, n_rows, n_cols);
        let reordered_scores: Vec<Vec<f32>> =
            as_row_major_matrix(constitutive_scores, n_rows, tissues_counts.len());

        let constitutive_donor_variances: Vec<f32> =
            as_row_major_matrix(sample_constitutive_scores, n_rows, n_cols)
                .into_iter()
                .map(|row| {
                    let tissue_variances: Vec<_> = column_indices_per_tissue
                        .iter()
                        .map(|(_, indices)| indices.iter().map(|&i| row[i]).collect::<Vec<f32>>())
                        .map(|donor_scores| calculate_standardized_variance(&donor_scores))
                        .collect();

                    calculate_standardized_variance(&tissue_variances)
                })
                .collect();

        let batch_scores: Vec<Score> = reordered_scores
            .iter()
            .zip(constitutive_donor_variances)
            .zip(reordered_counts)
            .zip(name_array)
            .map(
                |(((score_row, constitutive_donor_variance), counts_row), name)| Score {
                    key: name.unwrap().to_string(),
                    constitutive_score: calculate_median(&score_row),
                    constitutive_tissue_variance: calculate_standardized_variance(&score_row),
                    constitutive_donor_variance: constitutive_donor_variance,
                    n_expressed_tissues: score_row
                        .iter()
                        .filter(|&&score| score.is_finite())
                        .count(),
                    total_mean: calculate_mean(&counts_row),
                    tissue_mean: calculate_mean_over_group_medians(
                        &counts_row,
                        &column_indices_per_tissue,
                    ),
                    donor_mean: calculate_mean_over_group_medians(
                        &counts_row,
                        &column_indices_per_donor,
                    ),
                    total_variance: calculate_standardized_variance(&counts_row),
                    tissue_variance: calculate_variance_over_group_medians(
                        &counts_row,
                        &column_indices_per_tissue,
                    ),
                    donor_variance: calculate_variance_over_group_medians(
                        &counts_row,
                        &column_indices_per_donor,
                    ),
                },
            )
            .collect();

        scores.extend(batch_scores);
    }

    Ok(scores)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    let exon_metadata = ExonMetadataTable::new(args.exon_metadata)?;
    let donor_metadata = DonorMetadataTable::new(args.donor_metadata)?;

    let scores = read_exon_parquet(args.exon_counts, &exon_metadata, &donor_metadata)?;

    let mut writer = WriterBuilder::new()
        .delimiter(b'\t')
        .from_path(args.output)?;

    writer.write_record(&[
        "EXON_ID",
        "CHROM",
        "START",
        "END",
        "STRAND",
        "CONSTITUTIVE_SCORE",
        "CONSTITUTIVE_TISSUE_VARIANCE",
        "CONSTITUTIVE_DONOR_VARIANCE",
        "N_EXPRESSED_TISSUES",
        "TOTAL_MEAN",
        "TISSUE_MEAN",
        "DONOR_MEAN",
        "TOTAL_VARIANCE",
        "TISSUE_VARIANCE",
        "DONOR_VARIANCE",
    ])?;

    for score in scores {
        if let Some(region) = exon_metadata.get_exon(&score.key) {
            writer.write_record(&[
                score.key,
                region.chr.clone(),
                region.start.to_string(),
                region.end.to_string(),
                region.strand.clone(),
                score.constitutive_score.to_string(),
                score.constitutive_tissue_variance.to_string(),
                score.constitutive_donor_variance.to_string(),
                score.n_expressed_tissues.to_string(),
                score.total_mean.to_string(),
                score.tissue_mean.to_string(),
                score.donor_mean.to_string(),
                score.total_variance.to_string(),
                score.tissue_variance.to_string(),
                score.donor_variance.to_string(),
            ])?;
        }
    }

    Ok(())
}
