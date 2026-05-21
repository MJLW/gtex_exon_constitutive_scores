use serde::Deserialize;
use std::collections::HashMap;
use std::error::Error;
use std::fs::File;
use std::io::BufReader;
use std::path::PathBuf;

#[derive(Deserialize)]
struct MetadataRaw {
    exon_id: String,
    chr: String,

    #[serde(rename = "start_pos")]
    start: usize,

    #[serde(rename = "end_pos")]
    end: usize,
    strand: String,
}

#[derive(Debug, Deserialize)]
#[serde(from = "MetadataRaw")]
pub struct Metadata {
    exon_id: String,
    pub chr: String,
    pub start: usize,
    pub end: usize,
    pub size: usize,
    pub strand: String,
}

impl From<MetadataRaw> for Metadata {
    fn from(raw: MetadataRaw) -> Self {
        Self {
            exon_id: raw.exon_id,
            chr: raw.chr,
            start: raw.start,
            end: raw.end,
            size: raw.end - raw.start,
            strand: raw.strand,
        }
    }
}

pub struct MetadataTable {
    metadata: Vec<Metadata>,
    exon_id_map: HashMap<String, usize>,
}

impl MetadataTable {
    fn new(metadata: Vec<Metadata>) -> Self {
        let exon_id_map = metadata
            .iter()
            .enumerate()
            .map(|(i, m)| (m.exon_id.clone(), i))
            .collect();
        Self {
            metadata,
            exon_id_map,
        }
    }

    pub fn get_exon(&self, id: &str) -> Option<&Metadata> {
        self.exon_id_map.get(id).map(|&i| &self.metadata[i])
    }
}

pub fn read_metadata_tsv(path: PathBuf) -> Result<MetadataTable, Box<dyn Error>> {
    let mut reader = csv::ReaderBuilder::new()
        .delimiter(b'\t')
        .has_headers(true)
        .from_reader(BufReader::new(File::open(path)?));

    let records: Vec<Metadata> = reader
        .deserialize::<Metadata>()
        .collect::<Result<Vec<_>, _>>()?;

    Ok(MetadataTable::new(records))
}
