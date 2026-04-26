//! Tantivy-based full-text search index for Cerememory.
//!
//! Provides tokenized search across all stores, replacing O(n) substring scans
//! with an inverted index. Supports CJK via Tantivy's built-in tokenizer.

use std::sync::{Arc, Mutex};

use tantivy::collector::TopDocs;
use tantivy::query::QueryParser;
use tantivy::schema::*;
use tantivy::{Directory, Index, IndexReader, IndexWriter, ReloadPolicy, TantivyDocument};
use uuid::Uuid;

use cerememory_core::error::CerememoryError;
use cerememory_core::types::StoreType;

/// A hit from the full-text search index.
#[derive(Debug, Clone)]
pub struct TextSearchHit {
    pub record_id: Uuid,
    pub score: f32,
}

/// Fields stored in the Tantivy schema.
struct Fields {
    id: Field,
    store_type: Field,
    body: Field,
    summary: Field,
}

/// Global full-text search index spanning all memory stores.
pub struct TextIndex {
    index: Index,
    reader: IndexReader,
    writer: Arc<Mutex<IndexWriter>>,
    fields: Fields,
}

impl TextIndex {
    /// Open or create a file-backed index at the given path.
    pub fn open(path: &str) -> Result<Self, CerememoryError> {
        std::fs::create_dir_all(path)
            .map_err(|e| CerememoryError::Storage(format!("Failed to create index dir: {e}")))?;
        let dir = tantivy::directory::MmapDirectory::open(path)
            .map_err(|e| CerememoryError::Storage(format!("Tantivy dir open: {e}")))?;
        Self::open_with_dir(dir)
    }

    /// Create an in-memory index (for testing).
    pub fn open_in_memory() -> Result<Self, CerememoryError> {
        let dir = tantivy::directory::RamDirectory::create();
        Self::open_with_dir(dir)
    }

    fn open_with_dir(dir: impl Directory + 'static) -> Result<Self, CerememoryError> {
        let schema = Self::build_schema();
        let index = Index::open_or_create(dir, schema.clone())
            .map_err(|e| CerememoryError::Storage(format!("Tantivy index open: {e}")))?;

        let writer = index
            .writer(50_000_000) // 50MB heap
            .map_err(|e| CerememoryError::Storage(format!("Tantivy writer: {e}")))?;

        let reader = index
            .reader_builder()
            .reload_policy(ReloadPolicy::OnCommitWithDelay)
            .try_into()
            .map_err(|e| CerememoryError::Storage(format!("Tantivy reader: {e}")))?;

        let fields = Fields {
            id: schema.get_field("id").unwrap(),
            store_type: schema.get_field("store_type").unwrap(),
            body: schema.get_field("body").unwrap(),
            summary: schema.get_field("summary").unwrap(),
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
        builder.add_text_field("store_type", STRING | STORED);
        builder.add_text_field("body", TEXT);
        builder.add_text_field("summary", TEXT);
        builder.build()
    }

    fn build_document(
        &self,
        id: Uuid,
        store_type: StoreType,
        body: &str,
        summary: Option<&str>,
    ) -> TantivyDocument {
        let mut doc = TantivyDocument::new();
        doc.add_text(self.fields.id, id.to_string());
        doc.add_text(self.fields.store_type, store_type.to_string());
        doc.add_text(self.fields.body, body);
        if let Some(s) = summary {
            doc.add_text(self.fields.summary, s);
        }
        doc
    }

    fn lock_writer(&self) -> Result<std::sync::MutexGuard<'_, IndexWriter>, CerememoryError> {
        self.writer
            .lock()
            .map_err(|e| CerememoryError::Internal(format!("TextIndex writer lock: {e}")))
    }

    /// Add a document to the index.
    pub fn add(
        &self,
        id: Uuid,
        store_type: StoreType,
        body: &str,
        summary: Option<&str>,
    ) -> Result<(), CerememoryError> {
        let doc = self.build_document(id, store_type, body, summary);
        let mut writer = self.lock_writer()?;
        writer
            .add_document(doc)
            .map_err(|e| CerememoryError::Storage(format!("Tantivy add doc: {e}")))?;
        writer
            .commit()
            .map_err(|e| CerememoryError::Storage(format!("Tantivy commit: {e}")))?;
        Ok(())
    }

    /// Remove a document by record ID.
    pub fn remove(&self, id: Uuid) -> Result<(), CerememoryError> {
        let term = tantivy::Term::from_field_text(self.fields.id, &id.to_string());
        let mut writer = self.lock_writer()?;
        writer.delete_term(term);
        writer
            .commit()
            .map_err(|e| CerememoryError::Storage(format!("Tantivy commit: {e}")))?;
        Ok(())
    }

    /// Update a document (remove + re-add).
    pub fn update(
        &self,
        id: Uuid,
        store_type: StoreType,
        body: &str,
        summary: Option<&str>,
    ) -> Result<(), CerememoryError> {
        let term = tantivy::Term::from_field_text(self.fields.id, &id.to_string());
        let doc = self.build_document(id, store_type, body, summary);

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

    /// Search the index, optionally filtering by store types.
    pub fn search(
        &self,
        query: &str,
        stores: Option<&[StoreType]>,
        limit: usize,
    ) -> Result<Vec<TextSearchHit>, CerememoryError> {
        self.reader
            .reload()
            .map_err(|e| CerememoryError::Storage(format!("Tantivy reader reload: {e}")))?;

        let searcher = self.reader.searcher();
        let query_parser =
            QueryParser::for_index(&self.index, vec![self.fields.body, self.fields.summary]);

        let parsed = query_parser
            .parse_query(query)
            .map_err(|e| CerememoryError::Validation(format!("Invalid search query: {e}")))?;

        // Use larger multiplier when filtering by store to reduce missed results
        let search_limit = if stores.is_some() {
            limit * 5
        } else {
            limit * 2
        };
        let top_docs = searcher
            .search(&parsed, &TopDocs::with_limit(search_limit).order_by_score())
            .map_err(|e| CerememoryError::Storage(format!("Tantivy search: {e}")))?;

        let store_filter: Option<Vec<String>> =
            stores.map(|ss| ss.iter().map(|s| s.to_string()).collect());

        let mut hits = Vec::new();
        for (score, doc_address) in top_docs {
            let doc: TantivyDocument = searcher
                .doc(doc_address)
                .map_err(|e| CerememoryError::Storage(format!("Tantivy doc fetch: {e}")))?;

            // Filter by store type if requested
            if let Some(ref allowed) = store_filter {
                let st = doc
                    .get_first(self.fields.store_type)
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                if !allowed.iter().any(|a| a == st) {
                    continue;
                }
            }

            let id_str = doc
                .get_first(self.fields.id)
                .and_then(|v| v.as_str())
                .unwrap_or("");
            if let Ok(record_id) = Uuid::parse_str(id_str) {
                hits.push(TextSearchHit { record_id, score });
            }

            if hits.len() >= limit {
                break;
            }
        }

        Ok(hits)
    }

    /// Add multiple documents in a single write transaction (one lock, one commit).
    /// Uses upsert semantics: deletes any existing document with the same ID
    /// before adding the new one, preventing duplicates.
    pub fn add_batch(
        &self,
        records: &[(Uuid, StoreType, &str, Option<&str>)],
    ) -> Result<(), CerememoryError> {
        if records.is_empty() {
            return Ok(());
        }
        let mut writer = self.lock_writer()?;
        for (id, store_type, body, summary) in records {
            // Delete existing document first (upsert semantics)
            let term = tantivy::Term::from_field_text(self.fields.id, &id.to_string());
            writer.delete_term(term);

            let mut doc = TantivyDocument::new();
            doc.add_text(self.fields.id, id.to_string());
            doc.add_text(self.fields.store_type, store_type.to_string());
            doc.add_text(self.fields.body, body);
            if let Some(s) = summary {
                doc.add_text(self.fields.summary, s);
            }
            writer.add_document(doc).map_err(|e| {
                CerememoryError::Storage(format!("Failed to add document to text index: {e}"))
            })?;
        }
        writer
            .commit()
            .map_err(|e| CerememoryError::Storage(format!("Failed to commit text index: {e}")))?;
        Ok(())
    }

    /// Remove multiple documents in a single write transaction.
    pub fn remove_batch(&self, ids: &[Uuid]) -> Result<(), CerememoryError> {
        if ids.is_empty() {
            return Ok(());
        }
        let mut writer = self.lock_writer()?;
        for id in ids {
            let term = tantivy::Term::from_field_text(self.fields.id, &id.to_string());
            writer.delete_term(term);
        }
        writer.commit().map_err(|e| {
            CerememoryError::Storage(format!("Failed to commit text index removal: {e}"))
        })?;
        Ok(())
    }

    /// Rebuild the entire index from a set of records.
    pub fn rebuild(
        &self,
        records: &[(Uuid, StoreType, String, Option<String>)],
    ) -> Result<(), CerememoryError> {
        let mut writer = self.lock_writer()?;
        writer
            .delete_all_documents()
            .map_err(|e| CerememoryError::Storage(format!("Tantivy delete all: {e}")))?;

        for (id, store_type, body, summary) in records {
            let doc = self.build_document(*id, *store_type, body, summary.as_deref());
            writer
                .add_document(doc)
                .map_err(|e| CerememoryError::Storage(format!("Tantivy add doc: {e}")))?;
        }

        writer
            .commit()
            .map_err(|e| CerememoryError::Storage(format!("Tantivy commit: {e}")))?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add_and_search() {
        let idx = TextIndex::open_in_memory().unwrap();
        let id = Uuid::now_v7();
        idx.add(
            id,
            StoreType::Episodic,
            "The quick brown fox jumps over the lazy dog",
            None,
        )
        .unwrap();

        let hits = idx.search("quick", None, 10).unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].record_id, id);
        assert!(hits[0].score > 0.0);
    }

    #[test]
    fn search_with_summary() {
        let idx = TextIndex::open_in_memory().unwrap();
        let id = Uuid::now_v7();
        idx.add(
            id,
            StoreType::Semantic,
            "Detailed body text about Rust programming",
            Some("Rust programming facts"),
        )
        .unwrap();

        let hits = idx.search("Rust programming", None, 10).unwrap();
        assert!(!hits.is_empty());
        assert_eq!(hits[0].record_id, id);
    }

    #[test]
    fn remove_document() {
        let idx = TextIndex::open_in_memory().unwrap();
        let id = Uuid::now_v7();
        idx.add(id, StoreType::Episodic, "Temporary memory", None)
            .unwrap();

        idx.remove(id).unwrap();

        let hits = idx.search("Temporary", None, 10).unwrap();
        assert!(hits.is_empty());
    }

    #[test]
    fn store_filter() {
        let idx = TextIndex::open_in_memory().unwrap();
        let id1 = Uuid::now_v7();
        let id2 = Uuid::now_v7();

        idx.add(id1, StoreType::Episodic, "Event about cats", None)
            .unwrap();
        idx.add(id2, StoreType::Semantic, "Knowledge about cats", None)
            .unwrap();

        // Search only episodic
        let hits = idx
            .search("cats", Some(&[StoreType::Episodic]), 10)
            .unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].record_id, id1);

        // Search all stores
        let hits = idx.search("cats", None, 10).unwrap();
        assert_eq!(hits.len(), 2);
    }

    #[test]
    fn rebuild_index() {
        let idx = TextIndex::open_in_memory().unwrap();
        let id1 = Uuid::now_v7();
        let id2 = Uuid::now_v7();

        // Add one, then rebuild with different data
        idx.add(id1, StoreType::Episodic, "Old data", None).unwrap();

        idx.rebuild(&[
            (id1, StoreType::Episodic, "New body text".to_string(), None),
            (
                id2,
                StoreType::Semantic,
                "Second record".to_string(),
                Some("Summary".to_string()),
            ),
        ])
        .unwrap();

        // Old data gone
        let hits = idx.search("Old", None, 10).unwrap();
        assert!(hits.is_empty());

        // New data present
        let hits = idx.search("New body", None, 10).unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].record_id, id1);

        let hits = idx.search("Second", None, 10).unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].record_id, id2);
    }

    #[test]
    fn update_document() {
        let idx = TextIndex::open_in_memory().unwrap();
        let id = Uuid::now_v7();

        idx.add(id, StoreType::Episodic, "Original content", None)
            .unwrap();
        idx.update(id, StoreType::Episodic, "Updated content", None)
            .unwrap();

        let hits = idx.search("Original", None, 10).unwrap();
        assert!(hits.is_empty());

        let hits = idx.search("Updated", None, 10).unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].record_id, id);
    }

    #[test]
    fn empty_query_or_no_results() {
        let idx = TextIndex::open_in_memory().unwrap();
        let id = Uuid::now_v7();
        idx.add(id, StoreType::Episodic, "Some text here", None)
            .unwrap();

        let hits = idx.search("nonexistent", None, 10).unwrap();
        assert!(hits.is_empty());
    }

    #[test]
    fn multi_word_search() {
        let idx = TextIndex::open_in_memory().unwrap();
        let id = Uuid::now_v7();
        idx.add(
            id,
            StoreType::Episodic,
            "The quick brown fox jumps over the lazy dog",
            None,
        )
        .unwrap();

        let hits = idx.search("quick brown", None, 10).unwrap();
        assert!(!hits.is_empty());
        assert_eq!(hits[0].record_id, id);
    }

    // --- Batch tests ---

    #[test]
    fn add_batch_commit() {
        let idx = TextIndex::open_in_memory().unwrap();
        let id1 = Uuid::now_v7();
        let id2 = Uuid::now_v7();

        idx.add_batch(&[
            (id1, StoreType::Episodic, "First batch record", None),
            (
                id2,
                StoreType::Semantic,
                "Second batch record",
                Some("summary"),
            ),
        ])
        .unwrap();

        // Both should be searchable
        let hits = idx.search("batch record", None, 10).unwrap();
        assert_eq!(hits.len(), 2);
    }

    #[test]
    fn add_batch_empty() {
        let idx = TextIndex::open_in_memory().unwrap();
        // Empty batch should be a no-op without error
        idx.add_batch(&[]).unwrap();
    }

    #[test]
    fn add_batch_searchable() {
        let idx = TextIndex::open_in_memory().unwrap();
        let id1 = Uuid::now_v7();
        let id2 = Uuid::now_v7();
        let id3 = Uuid::now_v7();

        idx.add_batch(&[
            (id1, StoreType::Episodic, "Apples are delicious fruit", None),
            (id2, StoreType::Semantic, "Bananas contain potassium", None),
            (id3, StoreType::Episodic, "Cherries grow on trees", None),
        ])
        .unwrap();

        let hits = idx.search("potassium", None, 10).unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].record_id, id2);

        // Filter by store type
        let hits = idx
            .search(
                "delicious OR potassium OR trees",
                Some(&[StoreType::Episodic]),
                10,
            )
            .unwrap();
        assert_eq!(hits.len(), 2);
    }

    #[test]
    fn remove_batch() {
        let idx = TextIndex::open_in_memory().unwrap();
        let id1 = Uuid::now_v7();
        let id2 = Uuid::now_v7();
        let id3 = Uuid::now_v7();

        idx.add_batch(&[
            (id1, StoreType::Episodic, "Alpha content", None),
            (id2, StoreType::Episodic, "Beta content", None),
            (id3, StoreType::Episodic, "Gamma content", None),
        ])
        .unwrap();

        // Remove two in a single batch
        idx.remove_batch(&[id1, id2]).unwrap();

        // Only id3 should remain
        let hits = idx.search("content", None, 10).unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].record_id, id3);
    }

    #[test]
    fn remove_batch_empty() {
        let idx = TextIndex::open_in_memory().unwrap();
        // Empty remove batch should be a no-op without error
        idx.remove_batch(&[]).unwrap();
    }
}
