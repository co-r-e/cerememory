//! Cerememory evolution engine — self-tuning decay and recall parameters.
//!
//! Accumulates statistics from decay ticks and recall operations, then
//! applies rule-based adjustments to keep the system performing optimally.
//! All adjustments are capped at ±50% of the default value.

use std::collections::HashMap;
use std::sync::Mutex;

use cerememory_core::protocol::{EvolutionMetrics, ParameterAdjustment};
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

/// Rolling average with a fixed window size.
struct RollingAverage {
    values: Vec<f64>,
    window: usize,
}

impl RollingAverage {
    fn new(window: usize) -> Self {
        Self {
            values: Vec::new(),
            window,
        }
    }

    fn push(&mut self, value: f64) {
        self.values.push(value);
        if self.values.len() > self.window {
            self.values.remove(0);
        }
    }

    fn average(&self) -> Option<f64> {
        if self.values.is_empty() {
            return None;
        }
        Some(self.values.iter().sum::<f64>() / self.values.len() as f64)
    }

    fn len(&self) -> usize {
        self.values.len()
    }
}

/// Fidelity histogram with 10 buckets (0.0-0.1, 0.1-0.2, ..., 0.9-1.0).
struct FidelityHistogram {
    buckets: [u64; 10],
    total: u64,
}

impl FidelityHistogram {
    fn new() -> Self {
        Self {
            buckets: [0; 10],
            total: 0,
        }
    }

    fn observe(&mut self, fidelity: f64) {
        let idx = ((fidelity * 10.0).floor() as usize).min(9);
        self.buckets[idx] += 1;
        self.total += 1;
    }

    fn median_bucket(&self) -> f64 {
        if self.total == 0 {
            return 0.5;
        }
        let mut cumulative = 0u64;
        let half = self.total / 2;
        for (i, &count) in self.buckets.iter().enumerate() {
            cumulative += count;
            if cumulative > half {
                return (i as f64 + 0.5) / 10.0; // midpoint of bucket
            }
        }
        0.95 // all in last bucket
    }

    /// Fraction of records in the lowest bucket (0.0-0.1).
    fn lowest_bucket_fraction(&self) -> f64 {
        if self.total == 0 {
            return 0.0;
        }
        self.buckets[0] as f64 / self.total as f64
    }
}

struct EvolutionState {
    fidelity_histograms: HashMap<StoreType, FidelityHistogram>,
    recall_hit_rates: HashMap<StoreType, RollingAverage>,
    adjusted_params: HashMap<StoreType, StoreDecayDefaults>,
    adjustments: Vec<ParameterAdjustment>,
    detected_patterns: Vec<String>,
}

impl EvolutionState {
    fn new() -> Self {
        Self {
            fidelity_histograms: HashMap::new(),
            recall_hit_rates: HashMap::new(),
            adjusted_params: HashMap::new(),
            adjustments: Vec::new(),
            detected_patterns: Vec::new(),
        }
    }
}

/// Maximum adjustment factor (±50% of default value).
const MAX_ADJUSTMENT_FACTOR: f64 = 0.5;

/// Rolling average window for recall hit rates.
const RECALL_WINDOW: usize = 100;

/// Self-tuning evolution engine that adjusts decay and recall parameters
/// based on observed system behaviour.
pub struct EvolutionEngine {
    state: Mutex<EvolutionState>,
}

impl EvolutionEngine {
    pub fn new() -> Self {
        Self {
            state: Mutex::new(EvolutionState::new()),
        }
    }

    /// Get default decay parameters for a store type (static defaults).
    fn static_defaults(store_type: StoreType) -> StoreDecayDefaults {
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

    /// Get current (possibly adjusted) decay parameters for a store type.
    pub fn get_decay_defaults(&self, store_type: StoreType) -> StoreDecayDefaults {
        let state = self.state.lock().unwrap();
        state
            .adjusted_params
            .get(&store_type)
            .cloned()
            .unwrap_or_else(|| Self::static_defaults(store_type))
    }

    /// Observe fidelity scores from a decay tick for a specific store.
    pub fn observe_decay_tick(&self, store: StoreType, fidelity_scores: &[f64]) {
        let mut state = self.state.lock().unwrap();
        let histogram = state
            .fidelity_histograms
            .entry(store)
            .or_insert_with(FidelityHistogram::new);
        for &score in fidelity_scores {
            histogram.observe(score);
        }
        Self::evaluate(&mut state);
    }

    /// Observe recall hit rate for a specific store.
    /// hit_rate: fraction of recall results that had relevance > 0 (0.0-1.0).
    pub fn observe_recall(&self, store: StoreType, hit_rate: f64) {
        let mut state = self.state.lock().unwrap();
        let rolling = state
            .recall_hit_rates
            .entry(store)
            .or_insert_with(|| RollingAverage::new(RECALL_WINDOW));
        rolling.push(hit_rate);
        Self::evaluate(&mut state);
    }

    /// Get current evolution metrics.
    pub fn get_metrics(&self) -> EvolutionMetrics {
        let state = self.state.lock().unwrap();
        EvolutionMetrics {
            parameter_adjustments: state.adjustments.clone(),
            detected_patterns: state.detected_patterns.clone(),
            schema_adaptations: Vec::new(),
        }
    }

    /// Rule-based evaluation and parameter adjustment.
    fn evaluate(state: &mut EvolutionState) {
        // Clear previous adjustments and patterns
        state.adjustments.clear();
        state.detected_patterns.clear();

        for store_type in [
            StoreType::Episodic,
            StoreType::Semantic,
            StoreType::Procedural,
            StoreType::Emotional,
        ] {
            let defaults = Self::static_defaults(store_type);
            let mut adjusted = state
                .adjusted_params
                .get(&store_type)
                .cloned()
                .unwrap_or_else(|| defaults.clone());

            // Rule 1: If median fidelity < 0.3, decay is too aggressive — reduce exponent by 10%
            if let Some(histogram) = state.fidelity_histograms.get(&store_type) {
                let median = histogram.median_bucket();
                if median < 0.3 {
                    let new_val =
                        clamp_adjustment(adjusted.decay_exponent * 0.9, defaults.decay_exponent);
                    if (new_val - adjusted.decay_exponent).abs() > 1e-10 {
                        state.adjustments.push(ParameterAdjustment {
                            store: store_type,
                            parameter: "decay_exponent".to_string(),
                            original_value: defaults.decay_exponent,
                            current_value: new_val,
                            reason: format!(
                                "Median fidelity {median:.2} < 0.3: decay too aggressive"
                            ),
                        });
                        state.detected_patterns.push(format!(
                            "{store_type}: low median fidelity ({median:.2}), reducing decay"
                        ));
                        adjusted.decay_exponent = new_val;
                    }
                }

                // Rule 3: If >50% of records are in the lowest fidelity bucket, pruning may be too aggressive
                let lowest_frac = histogram.lowest_bucket_fraction();
                if lowest_frac > 0.5 {
                    let new_val = clamp_adjustment(
                        adjusted.prune_threshold * 0.9,
                        defaults.prune_threshold,
                    );
                    if (new_val - adjusted.prune_threshold).abs() > 1e-10 {
                        state.adjustments.push(ParameterAdjustment {
                            store: store_type,
                            parameter: "prune_threshold".to_string(),
                            original_value: defaults.prune_threshold,
                            current_value: new_val,
                            reason: format!(
                                "{:.0}% in lowest fidelity bucket: over-pruning",
                                lowest_frac * 100.0
                            ),
                        });
                        state.detected_patterns.push(format!(
                            "{store_type}: over-pruning detected ({:.0}% in lowest bucket)",
                            lowest_frac * 100.0
                        ));
                        adjusted.prune_threshold = new_val;
                    }
                }
            }

            // Rule 2: If recall hit rate < 0.2, retrieval is insufficient — increase boost by 10%
            if let Some(rolling) = state.recall_hit_rates.get(&store_type) {
                if let Some(avg_hit_rate) = rolling.average() {
                    if avg_hit_rate < 0.2 && rolling.len() >= 5 {
                        let new_val = clamp_adjustment(
                            adjusted.retrieval_boost * 1.1,
                            defaults.retrieval_boost,
                        );
                        if (new_val - adjusted.retrieval_boost).abs() > 1e-10 {
                            state.adjustments.push(ParameterAdjustment {
                                store: store_type,
                                parameter: "retrieval_boost".to_string(),
                                original_value: defaults.retrieval_boost,
                                current_value: new_val,
                                reason: format!(
                                    "Recall hit rate {avg_hit_rate:.2} < 0.2: retrieval insufficient"
                                ),
                            });
                            state.detected_patterns.push(format!(
                                "{store_type}: low recall hit rate ({avg_hit_rate:.2}), increasing boost"
                            ));
                            adjusted.retrieval_boost = new_val;
                        }
                    }
                }
            }

            state.adjusted_params.insert(store_type, adjusted);
        }
    }
}

/// Clamp an adjusted value to within ±50% of the default value.
fn clamp_adjustment(value: f64, default: f64) -> f64 {
    let min = default * (1.0 - MAX_ADJUSTMENT_FACTOR);
    let max = default * (1.0 + MAX_ADJUSTMENT_FACTOR);
    value.clamp(min, max)
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
    fn default_params_no_observations() {
        let engine = EvolutionEngine::new();
        let episodic = engine.get_decay_defaults(StoreType::Episodic);
        assert_eq!(episodic.decay_exponent, 0.3);
        assert_eq!(episodic.retrieval_boost, 1.5);
        assert_eq!(episodic.interference_rate, 0.1);
        assert_eq!(episodic.prune_threshold, 0.01);

        let semantic = engine.get_decay_defaults(StoreType::Semantic);
        assert!(semantic.decay_exponent < episodic.decay_exponent);
    }

    #[test]
    fn low_fidelity_reduces_exponent() {
        let engine = EvolutionEngine::new();
        let original = engine.get_decay_defaults(StoreType::Episodic).decay_exponent;

        // Feed many low-fidelity scores (all below 0.3)
        let low_scores: Vec<f64> = (0..100).map(|i| i as f64 * 0.002).collect(); // 0.0 to 0.198
        engine.observe_decay_tick(StoreType::Episodic, &low_scores);

        let adjusted = engine.get_decay_defaults(StoreType::Episodic).decay_exponent;
        assert!(
            adjusted < original,
            "Expected decay_exponent to decrease: original={original}, adjusted={adjusted}"
        );
    }

    #[test]
    fn high_fidelity_no_change() {
        let engine = EvolutionEngine::new();
        let original = engine.get_decay_defaults(StoreType::Episodic).decay_exponent;

        // Feed high-fidelity scores (all above 0.7)
        let high_scores: Vec<f64> = (0..100).map(|i| 0.7 + i as f64 * 0.003).collect();
        engine.observe_decay_tick(StoreType::Episodic, &high_scores);

        let adjusted = engine.get_decay_defaults(StoreType::Episodic).decay_exponent;
        assert_eq!(
            adjusted, original,
            "High fidelity should not change decay_exponent"
        );
    }

    #[test]
    fn low_recall_increases_boost() {
        let engine = EvolutionEngine::new();
        let original = engine.get_decay_defaults(StoreType::Semantic).retrieval_boost;

        // Feed low hit rates (need >= 5 observations)
        for _ in 0..10 {
            engine.observe_recall(StoreType::Semantic, 0.1);
        }

        let adjusted = engine.get_decay_defaults(StoreType::Semantic).retrieval_boost;
        assert!(
            adjusted > original,
            "Expected retrieval_boost to increase: original={original}, adjusted={adjusted}"
        );
    }

    #[test]
    fn adjustment_capped_50pct() {
        let engine = EvolutionEngine::new();
        let original_boost = EvolutionEngine::static_defaults(StoreType::Semantic).retrieval_boost;

        // Feed extreme low hit rates many times to try to push boost beyond 50%
        for _ in 0..500 {
            engine.observe_recall(StoreType::Semantic, 0.0);
        }

        let adjusted = engine.get_decay_defaults(StoreType::Semantic).retrieval_boost;
        let max_allowed = original_boost * 1.5;
        assert!(
            adjusted <= max_allowed + 1e-10,
            "Retrieval boost {adjusted} should not exceed 150% of default {max_allowed}"
        );
        assert!(
            adjusted >= original_boost * 0.5 - 1e-10,
            "Retrieval boost {adjusted} should not go below 50% of default"
        );
    }

    #[test]
    fn metrics_records_adjustments() {
        let engine = EvolutionEngine::new();

        // Initially no adjustments
        let metrics = engine.get_metrics();
        assert!(metrics.parameter_adjustments.is_empty());
        assert!(metrics.detected_patterns.is_empty());

        // Trigger an adjustment with low fidelity
        let low_scores: Vec<f64> = vec![0.05; 100];
        engine.observe_decay_tick(StoreType::Episodic, &low_scores);

        let metrics = engine.get_metrics();
        assert!(
            !metrics.parameter_adjustments.is_empty(),
            "Should have parameter adjustments after low fidelity"
        );
        assert!(
            !metrics.detected_patterns.is_empty(),
            "Should have detected patterns after low fidelity"
        );

        // Verify adjustment details
        let adj = &metrics.parameter_adjustments[0];
        assert_eq!(adj.store, StoreType::Episodic);
        assert!(!adj.reason.is_empty());
    }

    #[test]
    fn histogram_accumulates() {
        let mut histogram = FidelityHistogram::new();
        assert_eq!(histogram.total, 0);
        assert_eq!(histogram.median_bucket(), 0.5); // default when empty

        histogram.observe(0.05); // bucket 0
        histogram.observe(0.15); // bucket 1
        histogram.observe(0.95); // bucket 9

        assert_eq!(histogram.total, 3);
        assert_eq!(histogram.buckets[0], 1);
        assert_eq!(histogram.buckets[1], 1);
        assert_eq!(histogram.buckets[9], 1);
    }

    #[test]
    fn rolling_average_windowed() {
        let mut rolling = RollingAverage::new(3);
        assert_eq!(rolling.average(), None);
        assert_eq!(rolling.len(), 0);

        rolling.push(1.0);
        rolling.push(2.0);
        rolling.push(3.0);
        assert_eq!(rolling.len(), 3);
        assert!((rolling.average().unwrap() - 2.0).abs() < 1e-10);

        // Push a 4th value — oldest (1.0) should be evicted
        rolling.push(4.0);
        assert_eq!(rolling.len(), 3);
        assert!((rolling.average().unwrap() - 3.0).abs() < 1e-10); // (2+3+4)/3 = 3.0
    }

    #[test]
    fn over_pruning_detected() {
        let engine = EvolutionEngine::new();

        // Feed >50% scores in the lowest bucket (0.0-0.1)
        let mut scores = vec![0.05; 60]; // 60 in lowest bucket
        scores.extend(vec![0.5; 40]); // 40 in middle bucket
        engine.observe_decay_tick(StoreType::Procedural, &scores);

        let metrics = engine.get_metrics();
        let has_prune_adjustment = metrics
            .parameter_adjustments
            .iter()
            .any(|a| a.parameter == "prune_threshold" && a.store == StoreType::Procedural);
        assert!(
            has_prune_adjustment,
            "Should detect over-pruning when >50% in lowest bucket"
        );

        let has_pattern = metrics
            .detected_patterns
            .iter()
            .any(|p| p.contains("over-pruning"));
        assert!(has_pattern, "Should have over-pruning pattern detected");
    }

    #[test]
    fn multi_store_independent() {
        let engine = EvolutionEngine::new();

        // Trigger adjustment only for Episodic
        let low_scores: Vec<f64> = vec![0.05; 100];
        engine.observe_decay_tick(StoreType::Episodic, &low_scores);

        // Semantic should still have default values
        let semantic = engine.get_decay_defaults(StoreType::Semantic);
        let semantic_default = EvolutionEngine::static_defaults(StoreType::Semantic);
        assert_eq!(semantic.decay_exponent, semantic_default.decay_exponent);
        assert_eq!(semantic.retrieval_boost, semantic_default.retrieval_boost);

        // Episodic should be adjusted
        let episodic = engine.get_decay_defaults(StoreType::Episodic);
        let episodic_default = EvolutionEngine::static_defaults(StoreType::Episodic);
        assert!(episodic.decay_exponent < episodic_default.decay_exponent);
    }

    #[test]
    fn recall_requires_minimum_observations() {
        let engine = EvolutionEngine::new();
        let original = engine.get_decay_defaults(StoreType::Episodic).retrieval_boost;

        // Feed only 3 low hit rates (below the 5 minimum)
        for _ in 0..3 {
            engine.observe_recall(StoreType::Episodic, 0.1);
        }

        let adjusted = engine.get_decay_defaults(StoreType::Episodic).retrieval_boost;
        assert_eq!(
            adjusted, original,
            "Should not adjust with fewer than 5 recall observations"
        );

        // Now add 2 more to reach threshold
        for _ in 0..2 {
            engine.observe_recall(StoreType::Episodic, 0.1);
        }

        let adjusted = engine.get_decay_defaults(StoreType::Episodic).retrieval_boost;
        assert!(
            adjusted > original,
            "Should adjust after reaching 5 recall observations"
        );
    }
}
