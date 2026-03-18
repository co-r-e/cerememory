//! Shared helpers for integration tests.

#![allow(dead_code)]

use cerememory_core::protocol::*;
use cerememory_core::types::*;

pub fn text_content(text: &str) -> MemoryContent {
    MemoryContent {
        blocks: vec![ContentBlock {
            modality: Modality::Text,
            format: "text/plain".to_string(),
            data: text.as_bytes().to_vec(),
            embedding: None,
        }],
        summary: None,
    }
}

pub fn text_req(text: &str, store: StoreType) -> EncodeStoreRequest {
    EncodeStoreRequest {
        header: None,
        content: text_content(text),
        store: Some(store),
        emotion: None,
        context: None,
        associations: None,
    }
}

pub fn text_with_embedding(text: &str, store: StoreType, emb: Vec<f32>) -> EncodeStoreRequest {
    EncodeStoreRequest {
        header: None,
        content: MemoryContent {
            blocks: vec![ContentBlock {
                modality: Modality::Text,
                format: "text/plain".to_string(),
                data: text.as_bytes().to_vec(),
                embedding: Some(emb),
            }],
            summary: None,
        },
        store: Some(store),
        emotion: None,
        context: None,
        associations: None,
    }
}
