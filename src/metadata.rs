use serde::Deserialize;
use std::collections::HashMap;
use std::error::Error;
use std::fs::File;
use std::io::BufReader;
use std::path::PathBuf;

#[derive(Deserialize)]
struct ExonMetadataRaw {
    exon_id: String,
    chr: String,

    #[serde(rename = "start_pos")]
    start: usize,

    #[serde(rename = "end_pos")]
    end: usize,
    strand: String,
}

#[derive(Debug, Deserialize)]
#[serde(from = "ExonMetadataRaw")]
pub struct ExonMetadata {
    exon_id: String,
    pub chr: String,
    pub start: usize,
    pub end: usize,
    pub size: usize,
    pub strand: String,
}

impl From<ExonMetadataRaw> for ExonMetadata {
    fn from(raw: ExonMetadataRaw) -> Self {
        Self {
            exon_id: raw.exon_id,
            chr: raw.chr,
            start: raw.start,
            end: raw.end + 1,
            size: raw.end - raw.start + 1,
            strand: raw.strand,
        }
    }
}

pub struct ExonMetadataTable {
    metadata: Vec<ExonMetadata>,
    exon_id_map: HashMap<String, usize>,
}

impl ExonMetadataTable {
    pub fn new(path: PathBuf) -> Result<Self, Box<dyn Error>> {
        let mut reader = csv::ReaderBuilder::new()
            .delimiter(b'\t')
            .has_headers(true)
            .from_reader(BufReader::new(File::open(path)?));

        let metadata: Vec<ExonMetadata> = reader
            .deserialize::<ExonMetadata>()
            .collect::<Result<Vec<_>, _>>()?;

        let exon_id_map = metadata
            .iter()
            .enumerate()
            .map(|(i, m)| (m.exon_id.clone(), i))
            .collect();

        Ok(Self {
            metadata,
            exon_id_map,
        })
    }

    pub fn get_exon(&self, id: &str) -> Option<&ExonMetadata> {
        self.exon_id_map.get(id).map(|&i| &self.metadata[i])
    }
}

#[derive(Deserialize)]
struct DonorMetadataRaw {
    #[serde(rename = "SAMPID")]
    sample_id: String,

    #[serde(rename = "SMTSD")]
    tissue: String,
}

#[derive(Debug, Deserialize)]
#[serde(from = "DonorMetadataRaw")]
pub struct DonorMetadata {
    sample_id: String,
    tissue: String,
    donor_id: String,
}

impl From<DonorMetadataRaw> for DonorMetadata {
    fn from(raw: DonorMetadataRaw) -> Self {
        let mut split_sample_id = raw.sample_id.splitn(3, '-');
        split_sample_id.next();
        let donor_id = split_sample_id.next().unwrap().to_string();

        Self {
            sample_id: raw.sample_id,
            tissue: raw.tissue,
            donor_id: donor_id,
        }
    }
}

pub struct DonorMetadataTable {
    sample_ids: Vec<String>,
    tissue_map: HashMap<String, Vec<usize>>,
    donor_map: HashMap<String, Vec<usize>>,
}

impl DonorMetadataTable {
    pub fn new(path: PathBuf) -> Result<Self, Box<dyn Error>> {
        let mut reader = csv::ReaderBuilder::new()
            .delimiter(b'\t')
            .has_headers(true)
            .from_reader(BufReader::new(File::open(path)?));

        let metadata: Vec<DonorMetadata> = reader
            .deserialize::<DonorMetadata>()
            .collect::<Result<Vec<_>, _>>()?;

        let mut sample_ids: Vec<String> = Vec::with_capacity(metadata.len());
        let mut tissue_map: HashMap<String, Vec<usize>> = HashMap::new();
        let mut donor_map: HashMap<String, Vec<usize>> = HashMap::new();

        for (i, m) in metadata.iter().enumerate() {
            sample_ids.push(m.sample_id.to_owned());
            tissue_map.entry(m.tissue.to_owned()).or_default().push(i);
            donor_map.entry(m.donor_id.to_owned()).or_default().push(i);
        }

        Ok(Self {
            sample_ids: sample_ids,
            tissue_map: tissue_map,
            donor_map: donor_map,
        })
    }

    pub fn get_tissues(&self) -> Vec<&str> {
        self.tissue_map.keys().map(|s| s.as_str()).collect()
    }

    pub fn get_tissue_samples(&self, tissue: &str) -> Result<Vec<&str>, Box<dyn Error>> {
        let indices = self
            .tissue_map
            .get(tissue)
            .ok_or(format!("Missing tissue {}", tissue))?;

        let samples = indices
            .iter()
            .map(|&i| self.sample_ids[i].as_str())
            .collect();

        Ok(samples)
    }

    pub fn get_donors(&self) -> Vec<&str> {
        self.donor_map.keys().map(|s| s.as_str()).collect()
    }

    pub fn get_donor_samples(&self, donor: &str) -> Result<Vec<&str>, Box<dyn Error>> {
        let indices = self
            .donor_map
            .get(donor)
            .ok_or(format!("Missing donor {}", donor))?;

        let samples = indices
            .iter()
            .map(|&i| self.sample_ids[i].as_str())
            .collect();

        Ok(samples)
    }
}
