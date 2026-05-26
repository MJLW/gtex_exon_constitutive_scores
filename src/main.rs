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
    total_mean: f32,
    tissue_mean: f32,
    donor_mean: f32,
    total_variance: f32,
    tissue_variance: f32,
    donor_variance: f32,
}

fn find_gene_boundaries(arr: &StringArray) -> Vec<usize> {
    let len = arr.len();
    let mut result: Vec<usize> = Vec::new();

    if len == 0 {
        return result;
    }

    result.push(0);
    let mut prev = arr.value(0);

    for i in 1..len {
        let current = arr.value(i);

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
    read_counts: &Float32Array,
    exon_sizes: &Vec<usize>,
    gene_boundaries: &Vec<(usize, usize)>,
) -> Vec<f32> {
    gene_boundaries
        .iter()
        .map(|(start, end)| {
            let gene_reads = read_counts.slice(*start, *end - *start);
            let lengths = &exon_sizes[*start..*end];

            // Calculate coverage (counts / region length)
            let coverages: Vec<f32> = gene_reads
                .iter()
                .zip(lengths)
                .map(|(x, l)| x.unwrap() / (*l as f32))
                .collect();

            // Divide scores by max coverage to get a ratio of how often it is expressed
            // Vec<f32> does not have a iter.max() function, so fold into a pairwise max
            let max_coverage = coverages.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));

            // Play stupid games, win stupid prizes
            if max_coverage == 0.0 {
                return vec![0.0; end - start];
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
    let len = data.len();
    data.iter().sum::<f32>() / (len as f32)
}

fn calculate_median(mut data: Vec<f32>) -> f32 {
    data.sort_unstable_by(|a, b| a.total_cmp(b));
    let len = data.len();

    data[len / 2]
}

fn calculate_standardized_variance(data: &Vec<f32>) -> f32 {
    let len = data.len();
    let mean: f32 = data.iter().sum::<f32>() / (len as f32);

    (data.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / (len as f32))
        .div_checked(mean)
        .unwrap_or(0.0)
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
        .map(|scores| calculate_median(scores))
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
        .map(|scores| calculate_median(scores))
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

        // Gene symbol column
        let description_array = get_batch_column_as_array::<StringArray>(&batch, 0);

        // It processes until the last start, because that gene will likely be incomplete
        let gene_start_indices = find_gene_boundaries(&description_array);
        let n_rows = *gene_start_indices.last().unwrap();
        let gene_boundaries: Vec<(usize, usize)> = gene_start_indices
            .windows(2)
            .map(|pair| (pair[0], pair[1]))
            .collect();

        // Ensemble gene id + region index column
        let name_array = get_batch_column_as_array::<StringArray>(&batch, n_columns - 1);

        // Get exon sizes from metadata table
        let exon_sizes: Vec<usize> = name_array
            .iter()
            .map(|n| exon_metadata.get_exon(n.unwrap()).unwrap().size)
            .collect();

        // Process read counts
        let n_cols = n_columns - 2;
        let count_arrays: Vec<_> = (1..n_cols + 1)
            .into_iter()
            .map(|i| get_batch_column_as_array::<Float32Array>(&batch, i))
            .collect();

        let counts: Vec<f32> = count_arrays
            .iter()
            .map(|col| col.iter().map(|x| x.unwrap() as f32).collect::<Vec<f32>>())
            .flatten()
            .collect();

        // println!("{:?}", calculate_median(data)counts);

        let constitutive_scores: Vec<_> = count_arrays
            .into_iter()
            .map(|sample_counts| {
                calculate_constitutive_scores(sample_counts, &exon_sizes, &gene_boundaries)
            })
            .flatten()
            .collect();

        // Reorder scores such that samples are stored contiguously
        let reordered_counts: Vec<Vec<f32>> = as_row_major_matrix(counts, n_rows, n_cols);
        let reordered_scores: Vec<Vec<f32>> =
            as_row_major_matrix(constitutive_scores, n_rows, n_cols);

        // println!("{:?}", reordered_counts);

        let batch_scores: Vec<Score> = reordered_scores
            .iter()
            .zip(reordered_counts)
            .zip(name_array)
            .map(|((score_row, counts_row), name)| Score {
                key: name.unwrap().to_string(),
                constitutive_score: calculate_median(score_row.clone()),
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

        let counts: Vec<f32> = count_arrays
            .iter()
            .map(|col| col.iter().map(|x| x.unwrap() as f32).collect::<Vec<f32>>())
            .flatten()
            .collect();

        let constitutive_scores: Vec<_> = count_arrays
            .iter()
            .map(|read_counts| {
                calculate_constitutive_scores(
                    read_counts,
                    &exon_sizes,
                    &vec![(0, batch.num_rows())],
                )
            })
            .flatten()
            .collect();

        let reordered_counts: Vec<Vec<f32>> = as_row_major_matrix(counts, n_rows, n_cols);

        let reordered_scores: Vec<Vec<f32>> =
            as_row_major_matrix(constitutive_scores, n_rows, n_cols);

        let batch_scores: Vec<Score> = reordered_scores
            .iter()
            .zip(reordered_counts)
            .zip(name_array)
            .map(|((score_row, counts_row), name)| Score {
                key: name.unwrap().to_string(),
                constitutive_score: calculate_median(score_row.clone()),
                total_mean: calculate_mean(&counts_row),
                tissue_mean: calculate_mean_over_group_medians(
                    &counts_row,
                    &column_indices_per_tissue,
                ),
                donor_mean: calculate_mean_over_group_medians(
                    &counts_row,
                    &column_indices_per_donor,
                ),
                total_variance: calculate_standardized_variance(&score_row),
                tissue_variance: calculate_variance_over_group_medians(
                    &score_row,
                    &column_indices_per_tissue,
                ),
                donor_variance: calculate_variance_over_group_medians(
                    &score_row,
                    &column_indices_per_donor,
                ),
            })
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
