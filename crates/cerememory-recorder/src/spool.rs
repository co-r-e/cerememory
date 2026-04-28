use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};

use cerememory_core::protocol::{EncodeBatchStoreRawRequest, EncodeStoreRawRequest};
use uuid::Uuid;

use crate::{RecorderClient, RecorderError};

#[derive(Debug, Clone)]
pub struct Spool {
    dir: PathBuf,
}

impl Spool {
    pub fn new(dir: PathBuf) -> Self {
        Self { dir }
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    pub fn ensure_writable(&self) -> Result<(), RecorderError> {
        ensure_private_dir(&self.dir)?;
        let probe = self.dir.join(format!(".probe-{}", Uuid::now_v7()));
        {
            let mut file = create_private_file(&probe)?;
            file.write_all(b"ok")
                .map_err(|source| RecorderError::Spool {
                    path: probe.clone(),
                    source,
                })?;
        }
        fs::remove_file(&probe).map_err(|source| RecorderError::Spool {
            path: probe,
            source,
        })?;
        Ok(())
    }

    pub fn spool_batch(&self, records: &[EncodeStoreRawRequest]) -> Result<PathBuf, RecorderError> {
        if records.is_empty() {
            return Ok(self.dir.clone());
        }
        ensure_private_dir(&self.dir)?;

        let final_path = self.dir.join(format!(
            "batch-{}-{}.jsonl",
            chrono::Utc::now().timestamp_millis(),
            Uuid::now_v7()
        ));
        let temp_path = final_path.with_extension("jsonl.tmp");
        {
            let mut file = create_private_file(&temp_path)?;
            for record in records {
                serde_json::to_writer(&mut file, record)?;
                file.write_all(b"\n")
                    .map_err(|source| RecorderError::Spool {
                        path: temp_path.clone(),
                        source,
                    })?;
            }
            file.sync_all().map_err(|source| RecorderError::Spool {
                path: temp_path.clone(),
                source,
            })?;
        }
        fs::rename(&temp_path, &final_path).map_err(|source| RecorderError::Spool {
            path: final_path.clone(),
            source,
        })?;
        Ok(final_path)
    }

    pub async fn flush_pending(
        &self,
        client: &RecorderClient,
        max_batch_records: usize,
    ) -> Result<usize, RecorderError> {
        ensure_private_dir(&self.dir)?;

        let mut files = fs::read_dir(&self.dir)
            .map_err(|source| RecorderError::Spool {
                path: self.dir.clone(),
                source,
            })?
            .filter_map(Result::ok)
            .map(|entry| entry.path())
            .filter(|path| is_spool_batch_file(path))
            .collect::<Vec<_>>();
        files.sort();

        let mut flushed = 0usize;
        for file in files {
            let mut records = match read_spool_file(&file) {
                Ok(records) => records,
                Err(err) => {
                    let quarantine_path = quarantine_spool_file(&file)?;
                    eprintln!(
                        "cerememory-recorder: quarantined unreadable spool file {} to {} ({err})",
                        file.display(),
                        quarantine_path.display()
                    );
                    continue;
                }
            };
            while !records.is_empty() {
                let chunk_len = records.len().min(max_batch_records.max(1));
                client
                    .send_raw_batch(&EncodeBatchStoreRawRequest {
                        header: None,
                        records: records[..chunk_len].to_vec(),
                    })
                    .await?;
                records.drain(..chunk_len);
                flushed += chunk_len;
                if records.is_empty() {
                    fs::remove_file(&file).map_err(|source| RecorderError::Spool {
                        path: file.clone(),
                        source,
                    })?;
                } else {
                    rewrite_spool_file(&file, &records)?;
                }
            }
        }

        Ok(flushed)
    }
}

fn read_spool_file(path: &Path) -> Result<Vec<EncodeStoreRawRequest>, RecorderError> {
    let file = File::open(path).map_err(|source| RecorderError::Spool {
        path: path.to_path_buf(),
        source,
    })?;
    let reader = BufReader::new(file);
    let mut records = Vec::new();
    for line in reader.lines() {
        let line = line.map_err(|source| RecorderError::Spool {
            path: path.to_path_buf(),
            source,
        })?;
        if line.trim().is_empty() {
            continue;
        }
        records.push(serde_json::from_str::<EncodeStoreRawRequest>(&line)?);
    }
    Ok(records)
}

fn rewrite_spool_file(path: &Path, records: &[EncodeStoreRawRequest]) -> Result<(), RecorderError> {
    let temp_path = path.with_extension(format!("jsonl.rewrite-{}.tmp", Uuid::now_v7()));
    {
        let mut file = create_private_file(&temp_path)?;
        for record in records {
            serde_json::to_writer(&mut file, record)?;
            file.write_all(b"\n")
                .map_err(|source| RecorderError::Spool {
                    path: temp_path.clone(),
                    source,
                })?;
        }
        file.sync_all().map_err(|source| RecorderError::Spool {
            path: temp_path.clone(),
            source,
        })?;
    }
    fs::rename(&temp_path, path).map_err(|source| RecorderError::Spool {
        path: path.to_path_buf(),
        source,
    })
}

fn ensure_private_dir(path: &Path) -> Result<(), RecorderError> {
    let existed = path.try_exists().map_err(|source| RecorderError::Spool {
        path: path.to_path_buf(),
        source,
    })?;
    fs::create_dir_all(path).map_err(|source| RecorderError::Spool {
        path: path.to_path_buf(),
        source,
    })?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;

        let metadata = fs::symlink_metadata(path).map_err(|source| RecorderError::Spool {
            path: path.to_path_buf(),
            source,
        })?;
        if metadata.file_type().is_symlink() {
            return Err(RecorderError::Config(format!(
                "spool directory {} must not be a symlink",
                path.display()
            )));
        }

        if existed {
            let mode = metadata.permissions().mode() & 0o777;
            if mode & 0o077 != 0 {
                return Err(RecorderError::Config(format!(
                    "spool directory {} must be private before use; current mode is {:03o}, expected 0700 or stricter",
                    path.display(),
                    mode
                )));
            }
        } else {
            set_private_permissions(path)?;
        }
    }
    Ok(())
}

fn is_spool_batch_file(path: &Path) -> bool {
    let Some(file_name) = path.file_name().and_then(|name| name.to_str()) else {
        return false;
    };
    file_name.starts_with("batch-") && path.extension().is_some_and(|ext| ext == "jsonl")
}

#[cfg(unix)]
fn set_private_permissions(path: &Path) -> Result<(), RecorderError> {
    use std::os::unix::fs::PermissionsExt;
    fs::set_permissions(path, fs::Permissions::from_mode(0o700)).map_err(|source| {
        RecorderError::Spool {
            path: path.to_path_buf(),
            source,
        }
    })
}

#[cfg(not(unix))]
fn set_private_permissions(_path: &Path) -> Result<(), RecorderError> {
    Ok(())
}

fn create_private_file(path: &Path) -> Result<File, RecorderError> {
    let mut options = OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    options.open(path).map_err(|source| RecorderError::Spool {
        path: path.to_path_buf(),
        source,
    })
}

fn quarantine_spool_file(path: &Path) -> Result<PathBuf, RecorderError> {
    let quarantine_path = path.with_extension(format!("jsonl.bad-{}", Uuid::now_v7()));
    fs::rename(path, &quarantine_path).map_err(|source| RecorderError::Spool {
        path: quarantine_path.clone(),
        source,
    })?;
    Ok(quarantine_path)
}

#[cfg(test)]
fn make_private_dir_for_test(path: &Path) -> Result<(), RecorderError> {
    fs::create_dir_all(path).map_err(|source| RecorderError::Spool {
        path: path.to_path_buf(),
        source,
    })?;
    set_private_permissions(path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::capture::{capture_event_to_raw_request, CaptureEvent, CaptureEventType};
    use serde_json::json;

    #[test]
    fn spools_batch_as_jsonl() {
        let temp = tempfile::tempdir().unwrap();
        make_private_dir_for_test(temp.path()).unwrap();
        let spool = Spool::new(temp.path().to_path_buf());
        let req = capture_event_to_raw_request(CaptureEvent {
            session_id: "sess".to_string(),
            event_type: CaptureEventType::UserMessage,
            content: json!("hello"),
            turn_id: None,
            topic_id: None,
            action_id: None,
            source: None,
            speaker: None,
            timestamp: None,
            visibility: None,
            secrecy_level: None,
            metadata: None,
            meta: None,
        })
        .unwrap();

        let path = spool.spool_batch(&[req]).unwrap();
        assert!(path.exists());
        let records = read_spool_file(&path).unwrap();
        assert_eq!(records.len(), 1);
    }

    #[cfg(unix)]
    #[test]
    fn rejects_existing_non_private_spool_directory_without_chmoding_it() {
        use std::os::unix::fs::PermissionsExt;

        let temp = tempfile::tempdir().unwrap();
        fs::set_permissions(temp.path(), fs::Permissions::from_mode(0o755)).unwrap();
        let spool = Spool::new(temp.path().to_path_buf());

        let err = spool.ensure_writable().unwrap_err();
        assert!(err.to_string().contains("must be private"));
        let mode = std::fs::metadata(temp.path()).unwrap().permissions().mode() & 0o777;
        assert_eq!(mode, 0o755);
    }

    #[cfg(unix)]
    #[test]
    fn spool_files_and_directory_are_private() {
        use std::os::unix::fs::PermissionsExt;

        let temp = tempfile::tempdir().unwrap();
        let spool = Spool::new(temp.path().join("spool"));
        let req = capture_event_to_raw_request(CaptureEvent {
            session_id: "sess".to_string(),
            event_type: CaptureEventType::UserMessage,
            content: json!("hello"),
            turn_id: None,
            topic_id: None,
            action_id: None,
            source: None,
            speaker: None,
            timestamp: None,
            visibility: None,
            secrecy_level: None,
            metadata: None,
            meta: None,
        })
        .unwrap();

        let path = spool.spool_batch(&[req]).unwrap();
        let dir_mode = std::fs::metadata(spool.dir()).unwrap().permissions().mode() & 0o777;
        let file_mode = std::fs::metadata(path).unwrap().permissions().mode() & 0o777;

        assert_eq!(dir_mode, 0o700);
        assert_eq!(file_mode, 0o600);
    }

    #[test]
    fn identifies_only_recorder_batch_files_as_spool_files() {
        assert!(is_spool_batch_file(Path::new("batch-123.jsonl")));
        assert!(!is_spool_batch_file(Path::new("notes.jsonl")));
        assert!(!is_spool_batch_file(Path::new("batch-123.jsonl.tmp")));
    }
}
