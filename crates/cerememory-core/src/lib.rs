//! Core types, traits, and CMP protocol definitions for Cerememory.
//!
//! This crate defines the foundational data structures and interfaces
//! that all other Cerememory crates depend on. It contains:
//!
//! - Memory record types ([`MemoryRecord`], [`MemoryContent`], [`ContentBlock`])
//! - Fidelity and decay types ([`FidelityState`])
//! - Emotion representation ([`EmotionVector`])
//! - Association types ([`Association`], [`AssociationType`])
//! - Store type enumeration ([`StoreType`])
//! - CMP protocol request/response types
//! - Core traits ([`Store`], [`DecayEngine`], [`AssociationEngine`])

pub mod types;
pub mod protocol;
pub mod traits;
pub mod error;

pub use types::*;
pub use error::CerememoryError;
pub use protocol::*;
pub use traits::*;
