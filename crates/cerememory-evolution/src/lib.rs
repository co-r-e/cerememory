//! Cerememory evolution engine.
//!
//! Phase 1 stub: returns static default values.
//! Full self-tuning evolution engine planned for Phase 4.

use cerememory_core::protocol::EvolutionMetrics;
use cerememory_core::types::StoreType;
use serde::{Deserialize, Serialize};

/// Default decay parameters per store type.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoreDecayDefaults {
    pub decay_exponent: f64,
    pub retrieval_boost: f64,
    pub interference_rate: f64,
    pub prune_threshold: f64,
}

/// Stub evolution engine returning static defaults.
pub struct EvolutionEngine;

impl EvolutionEngine {
    pub fn new() -> Self {
        Self
    }

    /// Get default decay parameters for a store type.
    pub fn get_decay_defaults(&self, store_type: StoreType) -> StoreDecayDefaults {
        match store_type {
            StoreType::Episodic => StoreDecayDefaults {
                decay_exponent: 0.3,
                retrieval_boost: 1.5,
                interference_rate: 0.1,
                prune_threshold: 0.01,
            },
            StoreType::Semantic => StoreDecayDefaults {
                decay_exponent: 0.15,
                retrieval_boost: 2.0,
                interference_rate: 0.05,
                prune_threshold: 0.005,
            },
            StoreType::Procedural => StoreDecayDefaults {
                decay_exponent: 0.1,
                retrieval_boost: 2.5,
                interference_rate: 0.02,
                prune_threshold: 0.001,
            },
            StoreType::Emotional => StoreDecayDefaults {
                decay_exponent: 0.2,
                retrieval_boost: 1.8,
                interference_rate: 0.08,
                prune_threshold: 0.01,
            },
            StoreType::Working => StoreDecayDefaults {
                decay_exponent: 0.8,
                retrieval_boost: 1.0,
                interference_rate: 0.3,
                prune_threshold: 0.1,
            },
        }
    }

    /// Get current evolution metrics (stub: empty).
    pub fn get_metrics(&self) -> EvolutionMetrics {
        EvolutionMetrics::default()
    }
}

impl Default for EvolutionEngine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_params_are_reasonable() {
        let engine = EvolutionEngine::new();
        let episodic = engine.get_decay_defaults(StoreType::Episodic);
        assert_eq!(episodic.decay_exponent, 0.3);
        assert_eq!(episodic.retrieval_boost, 1.5);

        let semantic = engine.get_decay_defaults(StoreType::Semantic);
        assert!(semantic.decay_exponent < episodic.decay_exponent);
    }

    #[test]
    fn metrics_returns_defaults() {
        let engine = EvolutionEngine::new();
        let metrics = engine.get_metrics();
        assert!(metrics.parameter_adjustments.is_empty());
    }
}
