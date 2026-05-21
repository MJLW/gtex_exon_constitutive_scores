use clap::Parser;
use core::f32;
use std::{fs::File, path::PathBuf};

use arrow_array::{Array, Float32Array, RecordBatch, StringArray};
use arrow_select::concat::concat_batches;
use csv::WriterBuilder;
use parquet::arrow::arrow_reader::{ParquetRecordBatchReader, ParquetRecordBatchReaderBuilder};

mod metadata;
use crate::metadata::{MetadataTable, read_metadata_tsv};

#[derive(Debug, Parser)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(long)]
    exon_counts: PathBuf,

    #[arg(long)]
    exon_metadata: PathBuf,

    #[arg(long)]
    output: PathBuf,
}

struct Score {
    key: String,
    constitutive_score: f64,
    variance: f64,
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

fn prep_for_statrs(flat_data: Vec<f32>, n_rows: usize, n_cols: usize) -> Vec<Vec<f64>> {
    (0..n_rows)
        .into_iter()
        .map(|i| {
            (0..n_cols)
                .into_iter()
                .map(|j| flat_data[i + n_rows * j] as f64)
                .collect()
        })
        .collect()
}

// fn calculate_variance(data: &Vec<f64>) -> f64

// WARN: Made to work with GTEx v7, very particular to the column order and data types.
fn read_exon_parquet(
    path: PathBuf,
    metadata_table: &MetadataTable,
) -> Result<Vec<Score>, Box<dyn std::error::Error>> {
    let builder = ParquetRecordBatchReaderBuilder::try_new(File::open(path)?)?;
    let metadata = builder.metadata().file_metadata();
    let n_columns = metadata.schema_descr().num_columns() as usize;
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
            .map(|n| metadata_table.get_exon(n.unwrap()).unwrap().size)
            .collect();

        // Process read counts
        let n_cols = n_columns - 2;
        let constitutive_scores: Vec<_> = (1..n_cols + 1)
            .into_iter()
            .map(|i| get_batch_column_as_array::<Float32Array>(&batch, i))
            .map(|read_counts| {
                calculate_constitutive_scores(read_counts, &exon_sizes, &gene_boundaries)
            })
            .flatten()
            .collect();

        // Reorder scores such that samples are stored contiguously
        let reordered_scores: Vec<Vec<f64>> = prep_for_statrs(constitutive_scores, n_rows, n_cols);

        let batch_scores: Vec<Score> = reordered_scores
            .iter()
            .zip(name_array)
            .map(|(row, name)| {
                let mut sorted_row = row.clone();
                sorted_row.sort_unstable_by(|a, b| a.total_cmp(b));
                let len = sorted_row.len();
                let median = sorted_row[len / 2];

                let mean: f64 = row.iter().sum::<f64>() / (len as f64);
                let variance =
                    row.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / (len as f64);

                Score {
                    key: name.unwrap().to_string(),
                    constitutive_score: median,
                    variance: variance,
                }
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
            .map(|n| metadata_table.get_exon(n.unwrap()).unwrap().size)
            .collect();

        let n_rows = batch.num_rows();
        let n_cols = n_columns - 2;
        let constitutive_scores: Vec<_> = (1..n_cols + 1)
            .into_iter()
            .map(|i| get_batch_column_as_array::<Float32Array>(&batch, i))
            .map(|read_counts| {
                calculate_constitutive_scores(
                    read_counts,
                    &exon_sizes,
                    &vec![(0, batch.num_rows())],
                )
            })
            .flatten()
            .collect();

        let reordered_scores: Vec<Vec<f64>> = prep_for_statrs(constitutive_scores, n_rows, n_cols);

        let batch_scores: Vec<Score> = reordered_scores
            .iter()
            .zip(name_array)
            .map(|(row, name)| {
                let mut sorted_row = row.clone();
                sorted_row.sort_unstable_by(|a, b| a.total_cmp(b));
                let len = sorted_row.len();
                let median = sorted_row[len / 2];

                let mean: f64 = row.iter().sum::<f64>() / (len as f64);
                let variance =
                    row.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / (len as f64);

                Score {
                    key: name.unwrap().to_string(),
                    constitutive_score: median,
                    variance: variance,
                }
            })
            .collect();

        scores.extend(batch_scores);
    }

    Ok(scores)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    let metadata: MetadataTable = read_metadata_tsv(args.exon_metadata)?;

    let scores = read_exon_parquet(args.exon_counts, &metadata)?;

    let mut writer = WriterBuilder::new()
        .delimiter(b'\t')
        .from_path(args.output)?;

    for score in scores {
        if let Some(region) = metadata.get_exon(&score.key) {
            writer.write_record(&[
                score.key,
                region.chr.clone(),
                region.start.to_string(),
                region.end.to_string(),
                region.strand.clone(),
                score.constitutive_score.to_string(),
                score.variance.to_string(),
            ])?;
        }
    }

    Ok(())
}
