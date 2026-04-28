use std::collections::{HashSet, VecDeque};

use cerememory_core::protocol::{EncodeBatchStoreRawRequest, EncodeStoreRawRequest};
use tokio::io::{AsyncBufRead, AsyncBufReadExt, BufReader};

use crate::capture::{
    capture_event_dedupe_key, capture_event_to_raw_request, parse_capture_event_line,
};
use crate::{RecorderClient, RecorderConfig, RecorderError, Spool};

const DEDUPE_WINDOW: usize = 512;

#[derive(Debug, Default)]
pub struct IngestStats {
    pub accepted: usize,
    pub duplicate: usize,
    pub sent: usize,
    pub spooled: usize,
}

pub async fn run_ingest(config: RecorderConfig) -> Result<IngestStats, RecorderError> {
    let stdin = tokio::io::stdin();
    let reader = BufReader::new(stdin);
    ingest_reader(reader, config).await
}

pub async fn ingest_reader<R>(
    reader: R,
    config: RecorderConfig,
) -> Result<IngestStats, RecorderError>
where
    R: AsyncBufRead + Unpin,
{
    let client = RecorderClient::new(&config)?;
    let spool = Spool::new(config.spool_dir.clone());
    match spool.flush_pending(&client, config.batch_max_records).await {
        Ok(_) => {}
        Err(RecorderError::Send(err)) if err.retryable => {}
        Err(err) => return Err(err),
    }

    let mut stats = IngestStats::default();
    let mut batch = Vec::new();
    let mut lines = reader.lines();
    let mut interval = tokio::time::interval(config.flush_interval);
    let mut line_number = 0usize;
    let mut dedupe = RecentDedupe::new(DEDUPE_WINDOW);

    loop {
        tokio::select! {
            line = lines.next_line() => {
                let line = match line {
                    Ok(Some(line)) => line,
                    Ok(None) => {
                        flush_records(&client, &spool, &mut batch, &mut stats).await?;
                        break;
                    }
                    Err(err) => {
                        return Err(spool_pending_before_error(
                            &spool,
                            &mut batch,
                            &mut stats,
                            RecorderError::Io(err),
                        ));
                    }
                };
                line_number += 1;
                let trimmed = line.trim();
                if trimmed.is_empty() {
                    continue;
                }
                let event = match parse_capture_event_line(&line, line_number, config.max_event_bytes) {
                    Ok(event) => event,
                    Err(err) => {
                        return Err(spool_pending_before_error(
                            &spool,
                            &mut batch,
                            &mut stats,
                            err,
                        ));
                    }
                };
                if let Some(event) = event {
                    if let Some(key) = capture_event_dedupe_key(&event) {
                        if !dedupe.insert(&key) {
                            stats.duplicate += 1;
                            continue;
                        }
                    }
                    let record = match capture_event_to_raw_request(event) {
                        Ok(record) => record,
                        Err(err) => {
                            return Err(spool_pending_before_error(
                                &spool,
                                &mut batch,
                                &mut stats,
                                err,
                            ));
                        }
                    };
                    batch.push(record);
                    stats.accepted += 1;
                }
                if batch.len() >= config.batch_max_records {
                    flush_records(&client, &spool, &mut batch, &mut stats).await?;
                }
            }
            _ = interval.tick() => {
                flush_records(&client, &spool, &mut batch, &mut stats).await?;
            }
        }
    }

    Ok(stats)
}

fn spool_pending_before_error(
    spool: &Spool,
    batch: &mut Vec<EncodeStoreRawRequest>,
    stats: &mut IngestStats,
    cause: RecorderError,
) -> RecorderError {
    if batch.is_empty() {
        return cause;
    }

    match spool.spool_batch(batch) {
        Ok(path) => {
            stats.spooled += batch.len();
            batch.clear();
            eprintln!(
                "cerememory-recorder: input failed; spooled pending batch to {} before returning error ({cause})",
                path.display()
            );
            cause
        }
        Err(spool_err) => {
            eprintln!(
                "cerememory-recorder: input failed with {cause}; additionally failed to spool pending batch ({spool_err})"
            );
            spool_err
        }
    }
}

async fn flush_records(
    client: &RecorderClient,
    spool: &Spool,
    batch: &mut Vec<EncodeStoreRawRequest>,
    stats: &mut IngestStats,
) -> Result<(), RecorderError> {
    if batch.is_empty() {
        return Ok(());
    }

    let request = EncodeBatchStoreRawRequest {
        header: None,
        records: batch.clone(),
    };
    match client.send_raw_batch(&request).await {
        Ok(response) => {
            stats.sent += response.results.len();
            batch.clear();
            Ok(())
        }
        Err(send_error) => {
            let retryable = send_error.retryable;
            let path = spool.spool_batch(batch)?;
            stats.spooled += batch.len();
            batch.clear();
            eprintln!(
                "cerememory-recorder: send failed; spooled batch to {} ({send_error})",
                path.display()
            );
            if retryable {
                Ok(())
            } else {
                Err(RecorderError::Send(send_error))
            }
        }
    }
}

struct RecentDedupe {
    limit: usize,
    seen: HashSet<String>,
    order: VecDeque<String>,
}

impl RecentDedupe {
    fn new(limit: usize) -> Self {
        Self {
            limit,
            seen: HashSet::new(),
            order: VecDeque::new(),
        }
    }

    fn insert(&mut self, value: &str) -> bool {
        if self.seen.contains(value) {
            return false;
        }
        let value = value.to_string();
        self.seen.insert(value.clone());
        self.order.push_back(value);
        while self.order.len() > self.limit {
            if let Some(old) = self.order.pop_front() {
                self.seen.remove(&old);
            }
        }
        true
    }
}
