//! Shared helpers for integration tests.

#![allow(dead_code)]

use cerememory_core::protocol::*;
use cerememory_core::types::*;

pub fn text_content(text: &str) -> MemoryContent {
    text_content_with_embedding(text, None)
}

pub fn text_content_with_embedding(text: &str, embedding: Option<Vec<f32>>) -> MemoryContent {
    MemoryContent {
        blocks: vec![ContentBlock {
            modality: Modality::Text,
            format: "text/plain".to_string(),
            data: text.as_bytes().to_vec(),
            embedding,
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
        content: text_content_with_embedding(text, Some(emb)),
        store: Some(store),
        emotion: None,
        context: None,
        associations: None,
    }
}

pub fn text_with_emotion(
    text: &str,
    store: StoreType,
    emotion: EmotionVector,
) -> EncodeStoreRequest {
    EncodeStoreRequest {
        header: None,
        content: text_content(text),
        store: Some(store),
        emotion: Some(emotion),
        context: None,
        associations: None,
    }
}
