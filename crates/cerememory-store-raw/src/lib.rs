//! Cerememory raw journal store implementation backed by redb.
//!
//! The raw journal is an append-oriented preservation layer for verbatim
//! conversation turns, tool I/O, and other externally visible traces.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use chrono::{DateTime, Utc};
use redb::{Database, ReadableDatabase, ReadableTable, ReadableTableMetadata, TableDefinition};
use serde::{Deserialize, Serialize};
use tantivy::collector::TopDocs;
use tantivy::query::QueryParser;
use tantivy::schema::*;
use tantivy::{Directory, Index, IndexReader, IndexWriter, ReloadPolicy, TantivyDocument};
use uuid::Uuid;

use cerememory_core::error::CerememoryError;
use cerememory_core::types::RawJournalRecord;
use cerememory_store_common::{storage_err, StoreRecordCodec, StoreRecordMigrationStats};

/// Primary table: UUID (16 bytes) -> MessagePack-encoded `RawJournalRecord`.
const RAW_JOURNAL_RECORDS: TableDefinition<&[u8], &[u8]> =
    TableDefinition::new("raw_journal_records");
/// Secondary table: session id -> MessagePack-encoded ordered session index entries.
const RAW_JOURNAL_SESSION_INDEX: TableDefinition<&str, &[u8]> =
    TableDefinition::new("raw_journal_session_index");

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SessionIndexEntry {
    id: Uuid,
    created_at: DateTime<Utc>,
}

struct RawFields {
    id: Field,
    session_id: Field,
    body: Field,
}

#[derive(Clone)]
struct RawTextSearchHit {
    record_id: Uuid,
}

struct RawTextIndex {
    index: Index,
    reader: IndexReader,
    writer: Arc<Mutex<IndexWriter>>,
    fields: RawFields,
}

impl RawTextIndex {
    fn open(path: &Path) -> Result<Self, CerememoryError> {
        let dir_path = raw_text_index_path(path);
        std::fs::create_dir_all(&dir_path).map_err(|e| {
            CerememoryError::Storage(format!("Failed to create raw text index dir: {e}"))
        })?;
        let dir = tantivy::directory::MmapDirectory::open(&dir_path)
            .map_err(|e| CerememoryError::Storage(format!("Tantivy dir open: {e}")))?;
        Self::open_with_dir(dir)
    }

    fn open_in_memory() -> Result<Self, CerememoryError> {
        let dir = tantivy::directory::RamDirectory::create();
        Self::open_with_dir(dir)
    }

    fn open_with_dir(dir: impl Directory + 'static) -> Result<Self, CerememoryError> {
        let schema = Self::build_schema();
        let index = Index::open_or_create(dir, schema.clone())
            .map_err(|e| CerememoryError::Storage(format!("Tantivy index open: {e}")))?;
        let writer = index
            .writer(20_000_000)
            .map_err(|e| CerememoryError::Storage(format!("Tantivy writer: {e}")))?;
        let reader = index
            .reader_builder()
            .reload_policy(ReloadPolicy::OnCommitWithDelay)
            .try_into()
            .map_err(|e| CerememoryError::Storage(format!("Tantivy reader: {e}")))?;
        let fields = RawFields {
            id: schema.get_field("id").unwrap(),
            session_id: schema.get_field("session_id").unwrap(),
            body: schema.get_field("body").unwrap(),
        };

        Ok(Self {
            index,
            reader,
            writer: Arc::new(Mutex::new(writer)),
            fields,
        })
    }

    fn build_schema() -> Schema {
        let mut builder = Schema::builder();
        builder.add_text_field("id", STRING | STORED);
        builder.add_text_field("session_id", STRING | STORED);
        builder.add_text_field("body", TEXT);
        builder.build()
    }

    fn lock_writer(&self) -> Result<std::sync::MutexGuard<'_, IndexWriter>, CerememoryError> {
        self.writer
            .lock()
            .map_err(|e| CerememoryError::Internal(format!("RawTextIndex writer lock: {e}")))
    }

    fn add(&self, id: Uuid, session_id: &str, body: &str) -> Result<(), CerememoryError> {
        let mut doc = TantivyDocument::new();
        doc.add_text(self.fields.id, id.to_string());
        doc.add_text(self.fields.session_id, session_id);
        doc.add_text(self.fields.body, body);

        let mut writer = self.lock_writer()?;
        writer
            .add_document(doc)
            .map_err(|e| CerememoryError::Storage(format!("Tantivy add doc: {e}")))?;
        writer
            .commit()
            .map_err(|e| CerememoryError::Storage(format!("Tantivy commit: {e}")))?;
        Ok(())
    }

    fn remove(&self, id: Uuid) -> Result<(), CerememoryError> {
        let term = tantivy::Term::from_field_text(self.fields.id, &id.to_string());
        let mut writer = self.lock_writer()?;
        writer.delete_term(term);
        writer
            .commit()
            .map_err(|e| CerememoryError::Storage(format!("Tantivy commit: {e}")))?;
        Ok(())
    }

    fn update(&self, id: Uuid, session_id: &str, body: &str) -> Result<(), CerememoryError> {
        let term = tantivy::Term::from_field_text(self.fields.id, &id.to_string());
        let mut doc = TantivyDocument::new();
        doc.add_text(self.fields.id, id.to_string());
        doc.add_text(self.fields.session_id, session_id);
        doc.add_text(self.fields.body, body);

        let mut writer = self.lock_writer()?;
        writer.delete_term(term);
        writer
            .add_document(doc)
            .map_err(|e| CerememoryError::Storage(format!("Tantivy add doc: {e}")))?;
        writer
            .commit()
            .map_err(|e| CerememoryError::Storage(format!("Tantivy commit: {e}")))?;
        Ok(())
    }

    fn rebuild(&self, records: &[RawJournalRecord]) -> Result<(), CerememoryError> {
        let mut writer = self.lock_writer()?;
        writer
            .delete_all_documents()
            .map_err(|e| CerememoryError::Storage(format!("Tantivy delete all docs: {e}")))?;
        for record in records {
            if let Some(body) = record.text_content() {
                let mut doc = TantivyDocument::new();
                doc.add_text(self.fields.id, record.id.to_string());
                doc.add_text(self.fields.session_id, &record.session_id);
                doc.add_text(self.fields.body, body);
                writer
                    .add_document(doc)
                    .map_err(|e| CerememoryError::Storage(format!("Tantivy add doc: {e}")))?;
            }
        }
        writer
            .commit()
            .map_err(|e| CerememoryError::Storage(format!("Tantivy commit: {e}")))?;
        Ok(())
    }

    fn search(
        &self,
        query: &str,
        session_id: Option<&str>,
        limit: usize,
    ) -> Result<Vec<RawTextSearchHit>, CerememoryError> {
        self.reader
            .reload()
            .map_err(|e| CerememoryError::Storage(format!("Tantivy reader reload: {e}")))?;

        let searcher = self.reader.searcher();
        let query_parser = QueryParser::for_index(&self.index, vec![self.fields.body]);
        let parsed = query_parser
            .parse_query(query)
            .map_err(|e| CerememoryError::Validation(format!("Invalid search query: {e}")))?;

        let search_limit = if session_id.is_some() {
            limit * 5
        } else {
            limit * 3
        };
        let top_docs = searcher
            .search(
                &parsed,
                &TopDocs::with_limit(search_limit.max(limit)).order_by_score(),
            )
            .map_err(|e| CerememoryError::Storage(format!("Tantivy search: {e}")))?;

        let mut hits = Vec::new();
        for (score, doc_address) in top_docs {
            let doc: TantivyDocument = searcher
                .doc(doc_address)
                .map_err(|e| CerememoryError::Storage(format!("Tantivy doc fetch: {e}")))?;

            if let Some(session_filter) = session_id {
                let doc_session = doc
                    .get_first(self.fields.session_id)
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                if doc_session != session_filter {
                    continue;
                }
            }

            let id_str = doc
                .get_first(self.fields.id)
                .and_then(|v| v.as_str())
                .unwrap_or("");
            if let Ok(record_id) = Uuid::parse_str(id_str) {
                let _ = score;
                hits.push(RawTextSearchHit { record_id });
            }
            if hits.len() >= limit {
                break;
            }
        }
        Ok(hits)
    }
}

#[derive(Clone)]
pub struct RawJournalStore {
    db: Arc<Database>,
    text_index: Arc<RawTextIndex>,
    codec: StoreRecordCodec,
}

impl RawJournalStore {
    /// Open (or create) a persistent raw journal at `path`.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, CerememoryError> {
        Self::open_with_codec(path, StoreRecordCodec::plaintext())
    }

    /// Open (or create) a persistent raw journal at `path` with an explicit record codec.
    pub fn open_with_codec(
        path: impl AsRef<Path>,
        codec: StoreRecordCodec,
    ) -> Result<Self, CerememoryError> {
        Self::open_with_codec_and_text_index_persistence(path, codec, true)
    }

    /// Open (or create) a persistent raw journal with explicit record codec and text-index policy.
    pub fn open_with_codec_and_text_index_persistence(
        path: impl AsRef<Path>,
        codec: StoreRecordCodec,
        persist_text_index: bool,
    ) -> Result<Self, CerememoryError> {
        let db = Database::create(path.as_ref())
            .map_err(|e| CerememoryError::Storage(format!("Failed to open redb database: {e}")))?;
        let text_index = if persist_text_index {
            RawTextIndex::open(path.as_ref())?
        } else {
            RawTextIndex::open_in_memory()?
        };

        let store = Self {
            db: Arc::new(db),
            text_index: Arc::new(text_index),
            codec,
        };
        store.ensure_tables()?;
        Ok(store)
    }

    /// Create an ephemeral in-memory raw journal backed by a temporary file.
    pub fn open_in_memory() -> Result<Self, CerememoryError> {
        let tmp = tempfile::NamedTempFile::new()
            .map_err(|e| CerememoryError::Storage(format!("Failed to create temp file: {e}")))?;
        let path = tmp.into_temp_path();
        let _ = std::fs::remove_file(&path);
        let db = Database::create(&path).map_err(|e| {
            CerememoryError::Storage(format!("Failed to open in-memory redb database: {e}"))
        })?;
        let text_index = RawTextIndex::open_in_memory()?;

        let store = Self {
            db: Arc::new(db),
            text_index: Arc::new(text_index),
            codec: StoreRecordCodec::plaintext(),
        };
        store.ensure_tables()?;
        Ok(store)
    }

    fn ensure_tables(&self) -> Result<(), CerememoryError> {
        let txn = self.db.begin_write().map_err(storage_err)?;
        {
            let _ = txn.open_table(RAW_JOURNAL_RECORDS).map_err(storage_err)?;
            let _ = txn
                .open_table(RAW_JOURNAL_SESSION_INDEX)
                .map_err(storage_err)?;
        }
        txn.commit().map_err(storage_err)?;
        Ok(())
    }

    /// Rewrite legacy plaintext raw journal payloads using this store's encrypted codec.
    ///
    /// Already encrypted payloads are decoded with the active key to ensure the
    /// configured passphrase can actually read the whole journal.
    pub async fn migrate_plaintext_records_to_encrypted(
        &self,
    ) -> Result<StoreRecordMigrationStats, CerememoryError> {
        if !self.codec.encrypts_new_records() {
            return Err(CerememoryError::Validation(
                "store encryption passphrase is required before migrating raw journal records"
                    .to_string(),
            ));
        }

        let db = self.db.clone();
        let codec = self.codec.clone();
        tokio::task::spawn_blocking(move || {
            let txn = db.begin_write().map_err(storage_err)?;
            let mut stats = StoreRecordMigrationStats::empty();
            let mut rewrites: Vec<(Vec<u8>, Vec<u8>)> = Vec::new();

            {
                let records = txn.open_table(RAW_JOURNAL_RECORDS).map_err(storage_err)?;
                for entry in records.iter().map_err(storage_err)? {
                    let (key_guard, value_guard) = entry.map_err(storage_err)?;
                    let payload = value_guard.value();
                    let record: RawJournalRecord = codec.decode(payload)?;
                    stats.records_total += 1;

                    if StoreRecordCodec::is_encrypted_payload(payload) {
                        stats.records_already_encrypted += 1;
                    } else {
                        rewrites.push((key_guard.value().to_vec(), codec.encode(&record)?));
                        stats.records_migrated += 1;
                    }
                }
            }

            if !rewrites.is_empty() {
                let mut records = txn.open_table(RAW_JOURNAL_RECORDS).map_err(storage_err)?;
                for (key, payload) in rewrites {
                    records
                        .insert(key.as_slice(), payload.as_slice())
                        .map_err(storage_err)?;
                }
            }

            txn.commit().map_err(storage_err)?;
            Ok(stats)
        })
        .await
        .map_err(|e| CerememoryError::Internal(format!("spawn_blocking panicked: {e}")))?
    }

    /// Rebuild the raw journal text index from stored records.
    pub async fn rebuild_text_index(&self) -> Result<(), CerememoryError> {
        let records = self.get_all().await?;
        self.text_index.rebuild(&records)
    }

    /// Append a new raw journal record.
    pub async fn append(&self, record: RawJournalRecord) -> Result<Uuid, CerememoryError> {
        record.validate()?;

        let record_id = record.id;
        let text_payload = record
            .text_content()
            .map(|text| (record_id, record.session_id.clone(), text.to_string()));

        let db = self.db.clone();
        let codec = self.codec.clone();
        let _ = tokio::task::spawn_blocking(move || {
            let id = record.id;
            let packed = codec.encode(&record)?;

            let txn = db.begin_write().map_err(storage_err)?;
            {
                let mut records = txn.open_table(RAW_JOURNAL_RECORDS).map_err(storage_err)?;
                records
                    .insert(id.as_bytes().as_slice(), packed.as_slice())
                    .map_err(storage_err)?;

                let mut session_index = txn
                    .open_table(RAW_JOURNAL_SESSION_INDEX)
                    .map_err(storage_err)?;
                let mut entries: Vec<SessionIndexEntry> = match session_index
                    .get(record.session_id.as_str())
                    .map_err(storage_err)?
                {
                    Some(value_guard) => {
                        rmp_serde::from_slice(value_guard.value()).map_err(|e| {
                            CerememoryError::Serialization(format!(
                                "msgpack decode session index: {e}"
                            ))
                        })?
                    }
                    None => Vec::new(),
                };
                entries.push(SessionIndexEntry {
                    id,
                    created_at: record.created_at,
                });
                let packed_entries = rmp_serde::to_vec(&entries).map_err(|e| {
                    CerememoryError::Serialization(format!("msgpack encode session index: {e}"))
                })?;
                session_index
                    .insert(record.session_id.as_str(), packed_entries.as_slice())
                    .map_err(storage_err)?;
            }
            txn.commit().map_err(storage_err)?;
            Ok::<Uuid, CerememoryError>(id)
        })
        .await
        .map_err(|e| CerememoryError::Internal(format!("spawn_blocking panicked: {e}")))??;

        if let Some((id, session_id, text)) = text_payload {
            self.text_index.add(id, &session_id, &text)?;
        }
        Ok(record_id)
    }

    /// Update an existing raw journal record in place.
    pub async fn update(&self, record: RawJournalRecord) -> Result<(), CerememoryError> {
        record.validate()?;

        let previous = self
            .get(&record.id)
            .await?
            .ok_or_else(|| CerememoryError::RecordNotFound(record.id.to_string()))?;
        let previous_had_text = previous.text_content().is_some();
        let record_id = record.id;
        let text_payload = record
            .text_content()
            .map(|text| (record_id, record.session_id.clone(), text.to_string()));

        let previous_for_txn = previous.clone();
        let record_for_txn = record.clone();

        let db = self.db.clone();
        let codec = self.codec.clone();
        tokio::task::spawn_blocking(move || {
            let packed = codec.encode(&record_for_txn)?;

            let txn = db.begin_write().map_err(storage_err)?;
            {
                let mut records = txn.open_table(RAW_JOURNAL_RECORDS).map_err(storage_err)?;
                records
                    .insert(record_for_txn.id.as_bytes().as_slice(), packed.as_slice())
                    .map_err(storage_err)?;

                let mut session_index = txn
                    .open_table(RAW_JOURNAL_SESSION_INDEX)
                    .map_err(storage_err)?;

                let mut old_entries: Vec<SessionIndexEntry> = match session_index
                    .get(previous_for_txn.session_id.as_str())
                    .map_err(storage_err)?
                {
                    Some(value_guard) => {
                        rmp_serde::from_slice(value_guard.value()).map_err(|e| {
                            CerememoryError::Serialization(format!(
                                "msgpack decode session index: {e}"
                            ))
                        })?
                    }
                    None => Vec::new(),
                };
                old_entries.retain(|entry| entry.id != previous_for_txn.id);
                let packed_old_entries = rmp_serde::to_vec(&old_entries).map_err(|e| {
                    CerememoryError::Serialization(format!("msgpack encode session index: {e}"))
                })?;
                session_index
                    .insert(
                        previous_for_txn.session_id.as_str(),
                        packed_old_entries.as_slice(),
                    )
                    .map_err(storage_err)?;

                let mut new_entries: Vec<SessionIndexEntry> =
                    if previous_for_txn.session_id == record_for_txn.session_id {
                        old_entries
                    } else {
                        match session_index
                            .get(record_for_txn.session_id.as_str())
                            .map_err(storage_err)?
                        {
                            Some(value_guard) => rmp_serde::from_slice(value_guard.value())
                                .map_err(|e| {
                                    CerememoryError::Serialization(format!(
                                        "msgpack decode session index: {e}"
                                    ))
                                })?,
                            None => Vec::new(),
                        }
                    };
                new_entries.push(SessionIndexEntry {
                    id: record_for_txn.id,
                    created_at: record_for_txn.created_at,
                });
                let packed_new_entries = rmp_serde::to_vec(&new_entries).map_err(|e| {
                    CerememoryError::Serialization(format!("msgpack encode session index: {e}"))
                })?;
                session_index
                    .insert(
                        record_for_txn.session_id.as_str(),
                        packed_new_entries.as_slice(),
                    )
                    .map_err(storage_err)?;
            }
            txn.commit().map_err(storage_err)?;
            Ok::<(), CerememoryError>(())
        })
        .await
        .map_err(|e| CerememoryError::Internal(format!("spawn_blocking panicked: {e}")))??;

        if previous_had_text && text_payload.is_none() {
            self.text_index.remove(record_id)?;
        }
        if let Some((id, session_id, text)) = text_payload {
            if previous_had_text {
                self.text_index.update(id, &session_id, &text)?;
            } else {
                self.text_index.add(id, &session_id, &text)?;
            }
        }
        Ok(())
    }

    /// Retrieve a raw journal record by id.
    pub async fn get(&self, id: &Uuid) -> Result<Option<RawJournalRecord>, CerememoryError> {
        let db = self.db.clone();
        let id = *id;
        let codec = self.codec.clone();
        tokio::task::spawn_blocking(move || {
            let txn = db.begin_read().map_err(storage_err)?;
            let table = txn.open_table(RAW_JOURNAL_RECORDS).map_err(storage_err)?;
            get_raw_record_sync(&table, &id, &codec)
        })
        .await
        .map_err(|e| CerememoryError::Internal(format!("spawn_blocking panicked: {e}")))?
    }

    /// Delete a raw journal record by id.
    pub async fn delete(&self, id: &Uuid) -> Result<bool, CerememoryError> {
        let existing = self.get(id).await?;
        let Some(existing) = existing else {
            return Ok(false);
        };
        let existing_had_text = existing.text_content().is_some();
        let existing_session_id = existing.session_id.clone();
        let id = *id;
        let db = self.db.clone();
        tokio::task::spawn_blocking(move || {
            let txn = db.begin_write().map_err(storage_err)?;
            {
                let mut records = txn.open_table(RAW_JOURNAL_RECORDS).map_err(storage_err)?;
                let _ = records
                    .remove(id.as_bytes().as_slice())
                    .map_err(storage_err)?;

                let mut session_index = txn
                    .open_table(RAW_JOURNAL_SESSION_INDEX)
                    .map_err(storage_err)?;
                let mut entries: Vec<SessionIndexEntry> = match session_index
                    .get(existing_session_id.as_str())
                    .map_err(storage_err)?
                {
                    Some(value_guard) => {
                        rmp_serde::from_slice(value_guard.value()).map_err(|e| {
                            CerememoryError::Serialization(format!(
                                "msgpack decode session index: {e}"
                            ))
                        })?
                    }
                    None => Vec::new(),
                };
                entries.retain(|entry| entry.id != id);
                let packed_entries = rmp_serde::to_vec(&entries).map_err(|e| {
                    CerememoryError::Serialization(format!("msgpack encode session index: {e}"))
                })?;
                session_index
                    .insert(existing_session_id.as_str(), packed_entries.as_slice())
                    .map_err(storage_err)?;
            }
            txn.commit().map_err(storage_err)?;
            Ok::<bool, CerememoryError>(true)
        })
        .await
        .map_err(|e| CerememoryError::Internal(format!("spawn_blocking panicked: {e}")))??;

        if existing_had_text {
            self.text_index.remove(id)?;
        }
        Ok(true)
    }

    /// Return all raw journal records in storage.
    pub async fn get_all(&self) -> Result<Vec<RawJournalRecord>, CerememoryError> {
        let db = self.db.clone();
        let codec = self.codec.clone();
        tokio::task::spawn_blocking(move || {
            let txn = db.begin_read().map_err(storage_err)?;
            let table = txn.open_table(RAW_JOURNAL_RECORDS).map_err(storage_err)?;
            get_all_raw_records_sync(&table, &codec)
        })
        .await
        .map_err(|e| CerememoryError::Internal(format!("spawn_blocking panicked: {e}")))?
    }

    /// Return all records for a session, sorted by creation time.
    pub async fn query_session(
        &self,
        session_id: &str,
    ) -> Result<Vec<RawJournalRecord>, CerememoryError> {
        let session_id = session_id.trim().to_string();
        if session_id.is_empty() {
            return Err(CerememoryError::Validation(
                "session_id must not be empty".to_string(),
            ));
        }

        let db = self.db.clone();
        let codec = self.codec.clone();
        tokio::task::spawn_blocking(move || {
            let txn = db.begin_read().map_err(storage_err)?;
            let records_table = txn.open_table(RAW_JOURNAL_RECORDS).map_err(storage_err)?;
            let session_table = txn
                .open_table(RAW_JOURNAL_SESSION_INDEX)
                .map_err(storage_err)?;
            let entries = get_session_index_entries_sync(&session_table, &session_id)?;
            let mut records = get_raw_records_by_entries_sync(&records_table, &entries, &codec)?;
            records.sort_by(|a, b| {
                a.created_at
                    .cmp(&b.created_at)
                    .then_with(|| a.id.cmp(&b.id))
            });
            Ok(records)
        })
        .await
        .map_err(|e| CerememoryError::Internal(format!("spawn_blocking panicked: {e}")))?
    }

    /// Return all records for a session whose `created_at` falls within `[start, end]`.
    pub async fn query_session_range(
        &self,
        session_id: &str,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
    ) -> Result<Vec<RawJournalRecord>, CerememoryError> {
        let session_id = session_id.trim().to_string();
        if session_id.is_empty() {
            return Err(CerememoryError::Validation(
                "session_id must not be empty".to_string(),
            ));
        }
        if start > end {
            return Err(CerememoryError::Validation(
                "Invalid time range: start must be earlier than or equal to end".to_string(),
            ));
        }

        let db = self.db.clone();
        let codec = self.codec.clone();
        tokio::task::spawn_blocking(move || {
            let txn = db.begin_read().map_err(storage_err)?;
            let records_table = txn.open_table(RAW_JOURNAL_RECORDS).map_err(storage_err)?;
            let session_table = txn
                .open_table(RAW_JOURNAL_SESSION_INDEX)
                .map_err(storage_err)?;
            let entries = get_session_index_entries_sync(&session_table, &session_id)?;
            let filtered_entries: Vec<SessionIndexEntry> = entries
                .into_iter()
                .filter(|entry| entry.created_at >= start && entry.created_at <= end)
                .collect();
            let mut records =
                get_raw_records_by_entries_sync(&records_table, &filtered_entries, &codec)?;
            records.sort_by(|a, b| {
                a.created_at
                    .cmp(&b.created_at)
                    .then_with(|| a.id.cmp(&b.id))
            });
            Ok(records)
        })
        .await
        .map_err(|e| CerememoryError::Internal(format!("spawn_blocking panicked: {e}")))?
    }

    pub async fn count(&self) -> Result<usize, CerememoryError> {
        let db = self.db.clone();
        tokio::task::spawn_blocking(move || {
            let txn = db.begin_read().map_err(storage_err)?;
            let table = txn.open_table(RAW_JOURNAL_RECORDS).map_err(storage_err)?;
            let len = table.len().map_err(storage_err)?;
            usize::try_from(len).map_err(|_| {
                CerememoryError::Internal(format!("raw journal length {len} does not fit usize"))
            })
        })
        .await
        .map_err(|e| CerememoryError::Internal(format!("spawn_blocking panicked: {e}")))?
    }

    /// Search text content in the raw journal, optionally filtered to a session.
    pub async fn search_text(
        &self,
        query: &str,
        session_id: Option<&str>,
        limit: usize,
    ) -> Result<Vec<RawJournalRecord>, CerememoryError> {
        let hits = self.text_index.search(query, session_id, limit)?;
        let mut records = Vec::with_capacity(hits.len());
        for hit in hits {
            if let Some(record) = self.get(&hit.record_id).await? {
                records.push(record);
            }
        }
        Ok(records)
    }
}

fn raw_text_index_path(path: &Path) -> PathBuf {
    let stem = path
        .file_stem()
        .and_then(|stem| stem.to_str())
        .unwrap_or("raw_journal");
    let dir_name = format!("{stem}_text_index");
    path.with_file_name(dir_name)
}

fn get_raw_record_sync(
    table: &redb::ReadOnlyTable<&[u8], &[u8]>,
    id: &Uuid,
    codec: &StoreRecordCodec,
) -> Result<Option<RawJournalRecord>, CerememoryError> {
    match table.get(id.as_bytes().as_slice()).map_err(storage_err)? {
        Some(value_guard) => Ok(Some(codec.decode(value_guard.value())?)),
        None => Ok(None),
    }
}

fn get_all_raw_records_sync(
    table: &redb::ReadOnlyTable<&[u8], &[u8]>,
    codec: &StoreRecordCodec,
) -> Result<Vec<RawJournalRecord>, CerememoryError> {
    let mut records = Vec::new();
    for entry in table.iter().map_err(storage_err)? {
        let (_, value) = entry.map_err(storage_err)?;
        let record: RawJournalRecord = codec.decode(value.value())?;
        records.push(record);
    }
    Ok(records)
}

fn get_session_index_entries_sync(
    table: &redb::ReadOnlyTable<&str, &[u8]>,
    session_id: &str,
) -> Result<Vec<SessionIndexEntry>, CerememoryError> {
    match table.get(session_id).map_err(storage_err)? {
        Some(value_guard) => rmp_serde::from_slice(value_guard.value()).map_err(|e| {
            CerememoryError::Serialization(format!("msgpack decode session index: {e}"))
        }),
        None => Ok(Vec::new()),
    }
}

fn get_raw_records_by_entries_sync(
    table: &redb::ReadOnlyTable<&[u8], &[u8]>,
    entries: &[SessionIndexEntry],
    codec: &StoreRecordCodec,
) -> Result<Vec<RawJournalRecord>, CerememoryError> {
    let mut records = Vec::with_capacity(entries.len());
    for entry in entries {
        if let Some(record) = get_raw_record_sync(table, &entry.id, codec)? {
            records.push(record);
        }
    }
    Ok(records)
}

#[cfg(test)]
mod tests {
    use super::*;
    use cerememory_core::types::{RawSource, RawSpeaker, RawVisibility, SecrecyLevel};

    fn make_record(session_id: &str, text: &str) -> RawJournalRecord {
        RawJournalRecord::new_text(
            session_id,
            RawSource::Conversation,
            RawSpeaker::User,
            RawVisibility::Normal,
            SecrecyLevel::Public,
            text,
        )
    }

    #[tokio::test]
    async fn append_and_get_roundtrip() {
        let store = RawJournalStore::open_in_memory().unwrap();
        let record = make_record("sess-1", "hello raw journal");
        let id = record.id;

        store.append(record).await.unwrap();
        let restored = store.get(&id).await.unwrap().unwrap();

        assert_eq!(restored.id, id);
        assert_eq!(restored.session_id, "sess-1");
        assert_eq!(restored.text_content(), Some("hello raw journal"));
    }

    #[tokio::test]
    async fn encrypted_store_roundtrip_and_requires_key() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("raw_journal.redb");
        let store = RawJournalStore::open_with_codec(
            &path,
            StoreRecordCodec::encrypted_from_passphrase("correct horse battery staple").unwrap(),
        )
        .unwrap();
        let record = make_record("sess-1", "encrypted raw journal");
        let id = record.id;
        store.append(record).await.unwrap();
        drop(store);

        let reopened = RawJournalStore::open_with_codec(
            &path,
            StoreRecordCodec::encrypted_from_passphrase("correct horse battery staple").unwrap(),
        )
        .unwrap();
        assert_eq!(
            reopened.get(&id).await.unwrap().unwrap().text_content(),
            Some("encrypted raw journal")
        );
        drop(reopened);

        let plaintext_reopen = RawJournalStore::open(&path).unwrap();
        let err = plaintext_reopen.get(&id).await.unwrap_err();
        assert!(matches!(err, CerememoryError::Unauthorized(_)));
    }

    #[tokio::test]
    async fn encrypted_store_reads_legacy_plaintext_records() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("raw_journal.redb");
        let plaintext = RawJournalStore::open(&path).unwrap();
        let record = make_record("sess-1", "legacy raw journal");
        let id = record.id;
        plaintext.append(record).await.unwrap();
        drop(plaintext);

        let encrypted = RawJournalStore::open_with_codec(
            &path,
            StoreRecordCodec::encrypted_from_passphrase("new passphrase").unwrap(),
        )
        .unwrap();
        assert_eq!(
            encrypted.get(&id).await.unwrap().unwrap().text_content(),
            Some("legacy raw journal")
        );
    }

    #[tokio::test]
    async fn migrate_plaintext_records_to_encrypted_is_idempotent() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("raw_journal.redb");
        let plaintext = RawJournalStore::open(&path).unwrap();
        let record = make_record("sess-1", "legacy raw journal to migrate");
        let id = record.id;
        plaintext.append(record).await.unwrap();
        drop(plaintext);

        let encrypted = RawJournalStore::open_with_codec(
            &path,
            StoreRecordCodec::encrypted_from_passphrase("migration passphrase").unwrap(),
        )
        .unwrap();

        let stats = encrypted
            .migrate_plaintext_records_to_encrypted()
            .await
            .unwrap();
        assert_eq!(stats.records_total, 1);
        assert_eq!(stats.records_migrated, 1);
        assert_eq!(stats.records_already_encrypted, 0);
        assert_eq!(
            encrypted.get(&id).await.unwrap().unwrap().text_content(),
            Some("legacy raw journal to migrate")
        );
        assert_eq!(
            encrypted
                .query_session("sess-1")
                .await
                .unwrap()
                .first()
                .and_then(RawJournalRecord::text_content),
            Some("legacy raw journal to migrate")
        );

        let stats = encrypted
            .migrate_plaintext_records_to_encrypted()
            .await
            .unwrap();
        assert_eq!(stats.records_total, 1);
        assert_eq!(stats.records_migrated, 0);
        assert_eq!(stats.records_already_encrypted, 1);
        drop(encrypted);

        let plaintext_reopen = RawJournalStore::open(&path).unwrap();
        let err = plaintext_reopen.get(&id).await.unwrap_err();
        assert!(matches!(err, CerememoryError::Unauthorized(_)));
    }

    #[tokio::test]
    async fn query_session_filters_records() {
        let store = RawJournalStore::open_in_memory().unwrap();
        store.append(make_record("sess-1", "one")).await.unwrap();
        store.append(make_record("sess-2", "two")).await.unwrap();
        store.append(make_record("sess-1", "three")).await.unwrap();

        let sess_1 = store.query_session("sess-1").await.unwrap();
        assert_eq!(sess_1.len(), 2);
        assert!(sess_1.iter().all(|record| record.session_id == "sess-1"));
    }

    #[tokio::test]
    async fn query_session_range_filters_by_time() {
        let store = RawJournalStore::open_in_memory().unwrap();
        let base = Utc::now();

        let mut early = make_record("sess-1", "early");
        early.created_at = base - chrono::Duration::hours(2);
        early.updated_at = early.created_at;

        let mut middle = make_record("sess-1", "middle");
        middle.created_at = base - chrono::Duration::hours(1);
        middle.updated_at = middle.created_at;

        let mut late = make_record("sess-1", "late");
        late.created_at = base + chrono::Duration::hours(1);
        late.updated_at = late.created_at;

        store.append(early).await.unwrap();
        store.append(middle).await.unwrap();
        store.append(late).await.unwrap();

        let records = store
            .query_session_range(
                "sess-1",
                base - chrono::Duration::hours(1),
                base + chrono::Duration::minutes(10),
            )
            .await
            .unwrap();

        assert_eq!(records.len(), 1);
        assert_eq!(records[0].text_content(), Some("middle"));
    }

    #[tokio::test]
    async fn count_tracks_records() {
        let store = RawJournalStore::open_in_memory().unwrap();
        assert_eq!(store.count().await.unwrap(), 0);

        store.append(make_record("sess-1", "one")).await.unwrap();
        store.append(make_record("sess-1", "two")).await.unwrap();

        assert_eq!(store.count().await.unwrap(), 2);
    }

    #[tokio::test]
    async fn search_text_filters_by_query_and_session() {
        let store = RawJournalStore::open_in_memory().unwrap();
        store
            .append(make_record("sess-1", "timeout retries are idempotent only"))
            .await
            .unwrap();
        store
            .append(make_record("sess-2", "timeout budget differs"))
            .await
            .unwrap();

        let sess_1 = store
            .search_text("idempotent", Some("sess-1"), 10)
            .await
            .unwrap();
        assert_eq!(sess_1.len(), 1);
        assert_eq!(sess_1[0].session_id, "sess-1");

        let all = store.search_text("timeout", None, 10).await.unwrap();
        assert_eq!(all.len(), 2);
    }
}
