//! Memory decay engine for Cerememory.
//!
//! Implements the modified power-law decay model with emotional modulation
//! and spaced repetition reinforcement. See ADR-005 for the mathematical model.
//!
//! # Mathematical Model
//!
//! **Fidelity decay:**
//! ```text
//! F(t) = F_0 * (1 + t/S)^(-d) * E_mod
//! ```
//!
//! **Noise accumulation:**
//! ```text
//! N(t) = N_0 + interference_rate * sqrt(t) * (1 - F(t))
//! ```
//!
//! **Stability update (on retrieval):**
//! ```text
//! S_new = S_old * (1 + retrieval_boost * S_old^(-0.2))
//! ```

pub mod math;

use cerememory_core::{DecayEngine, DecayInput, DecayOutput, DecayTickResult, FidelityState};
use chrono::Utc;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

/// Configuration parameters for the power-law decay engine.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecayParams {
    /// Exponent `d` in the power-law decay formula (default: 0.3).
    pub decay_exponent: f64,
    /// Boost factor applied to stability on each retrieval (default: 1.5).
    pub retrieval_boost: f64,
    /// Rate of noise accumulation from interference (default: 0.1).
    pub interference_rate: f64,
    /// Fidelity threshold below which a record is eligible for pruning (default: 0.01).
    pub prune_threshold: f64,
    /// Initial fidelity for new records (default: 1.0).
    pub initial_fidelity: f64,
}

impl Default for DecayParams {
    fn default() -> Self {
        Self {
            decay_exponent: 0.3,
            retrieval_boost: 1.5,
            interference_rate: 0.1,
            prune_threshold: 0.01,
            initial_fidelity: 1.0,
        }
    }
}

/// Power-law decay engine implementing the `DecayEngine` trait.
///
/// Processes records in parallel using rayon. Each record is independently
/// computed, so the engine holds no mutable state and is safe for concurrent use.
#[derive(Clone)]
pub struct PowerLawDecayEngine {
    params: DecayParams,
}

impl PowerLawDecayEngine {
    /// Create a new decay engine with the given parameters.
    pub fn new(params: DecayParams) -> Self {
        Self { params }
    }

    /// Create a new decay engine with default parameters.
    pub fn with_defaults() -> Self {
        Self::new(DecayParams::default())
    }

    /// Return a reference to the current parameters.
    pub fn params(&self) -> &DecayParams {
        &self.params
    }

    /// Compute the stability boost for a retrieval event.
    ///
    /// This is exposed publicly so that callers (e.g., the hippocampal coordinator)
    /// can apply stability boosts when records are accessed.
    pub fn boost_stability(&self, current_stability: f64) -> f64 {
        math::compute_stability_boost(current_stability, self.params.retrieval_boost)
    }

    /// Compute decay for a single record.
    /// Uses the later of last_accessed_at and last_decay_tick as baseline,
    /// so only the delta since the last tick (or access) is applied.
    fn compute_single(&self, input: &DecayInput, now_secs_since_epoch: f64) -> DecayOutput {
        let last_access_secs = input.last_accessed_at.timestamp() as f64;
        let last_tick_secs = input.fidelity.last_decay_tick.timestamp() as f64;
        let baseline_secs = last_access_secs.max(last_tick_secs);
        let t_secs = (now_secs_since_epoch - baseline_secs).max(0.0);

        let f0 = input.fidelity.score;
        let n0 = input.fidelity.noise_level;
        let stability = input.fidelity.stability.max(f64::MIN_POSITIVE);
        let decay_exponent = input.fidelity.decay_rate; // per-record exponent
        let emotion_mod = math::compute_emotion_mod(input.emotion.intensity);

        let new_score = math::compute_fidelity(f0, t_secs, stability, decay_exponent, emotion_mod);
        let new_noise = math::compute_noise(n0, t_secs, new_score, self.params.interference_rate);

        let should_prune = new_score < self.params.prune_threshold;

        let now = Utc::now();
        DecayOutput {
            id: input.id,
            new_fidelity: FidelityState {
                score: new_score,
                noise_level: new_noise,
                decay_rate: decay_exponent,
                emotional_anchor: emotion_mod,
                reinforcement_count: input.fidelity.reinforcement_count,
                stability,
                last_decay_tick: now,
            },
            should_prune,
        }
    }
}

impl DecayEngine for PowerLawDecayEngine {
    fn compute_tick(&self, records: &[DecayInput], tick_duration_secs: f64) -> DecayTickResult {
        // Advance each record by whichever is larger:
        // - actual wall-clock time since the baseline
        // - the explicit tick duration
        //
        // This preserves decay for very old records while avoiding the previous
        // double-counting of "wall clock elapsed + tick duration".
        let wall_clock_now = Utc::now().timestamp() as f64;
        let effective_now = records
            .iter()
            .map(|input| {
                let baseline = input
                    .last_accessed_at
                    .timestamp()
                    .max(input.fidelity.last_decay_tick.timestamp())
                    as f64;
                wall_clock_now.max(baseline + tick_duration_secs)
            })
            .collect::<Vec<_>>();

        let updates: Vec<DecayOutput> = records
            .par_iter()
            .zip(effective_now.into_par_iter())
            .map(|(input, effective_now)| self.compute_single(input, effective_now))
            .collect();

        let records_updated = updates.len() as u32;
        let records_below_threshold = updates
            .iter()
            .filter(|o| o.new_fidelity.score < self.params.prune_threshold)
            .count() as u32;
        let records_pruned = updates.iter().filter(|o| o.should_prune).count() as u32;

        DecayTickResult {
            updates,
            records_updated,
            records_below_threshold,
            records_pruned,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cerememory_core::{EmotionVector, FidelityState};
    use chrono::{Duration, Utc};
    use uuid::Uuid;

    fn make_input(
        mut fidelity: FidelityState,
        emotion: EmotionVector,
        last_accessed_at: chrono::DateTime<Utc>,
    ) -> DecayInput {
        // Align last_decay_tick with last_accessed_at so delta-based decay works correctly
        fidelity.last_decay_tick = last_accessed_at;
        DecayInput {
            id: Uuid::now_v7(),
            fidelity,
            emotion,
            last_accessed_at,
            access_count: 0,
        }
    }

    fn default_fidelity() -> FidelityState {
        FidelityState::default()
    }

    fn default_emotion() -> EmotionVector {
        EmotionVector::default()
    }

    // ── Test 1: F(0) = 1.0 when no time has elapsed ──────────────────────

    #[test]
    fn fidelity_unchanged_when_no_time_elapsed() {
        let engine = PowerLawDecayEngine::with_defaults();
        let now = Utc::now();
        let input = make_input(default_fidelity(), default_emotion(), now);

        let result = engine.compute_tick(&[input], 0.0);
        assert_eq!(result.updates.len(), 1);

        let f = result.updates[0].new_fidelity.score;
        // Allow a tiny window for the sub-second between input creation and compute.
        assert!(
            f > 0.99,
            "Fidelity should be ~1.0 when just created, got {f}"
        );
    }

    // ── Test 2: Meaningful decrease after 3600s ──────────────────────────

    #[test]
    fn fidelity_decreases_after_1h() {
        let engine = PowerLawDecayEngine::with_defaults();
        let one_hour_ago = Utc::now() - Duration::hours(1);
        let input = make_input(default_fidelity(), default_emotion(), one_hour_ago);

        let result = engine.compute_tick(&[input], 3600.0);
        let f = result.updates[0].new_fidelity.score;
        assert!(
            f < 0.5,
            "After 1 hour with default params, fidelity should be well below 0.5, got {f}"
        );
        assert!(f > 0.0, "Fidelity should still be positive, got {f}");
    }

    #[test]
    fn tick_uses_explicit_duration_without_double_counting() {
        let engine = PowerLawDecayEngine::with_defaults();
        let now = Utc::now();
        let input = make_input(default_fidelity(), default_emotion(), now);

        let result = engine.compute_tick(std::slice::from_ref(&input), 3600.0);
        let expected = math::compute_fidelity(
            input.fidelity.score,
            3600.0,
            input.fidelity.stability,
            input.fidelity.decay_rate,
            math::compute_emotion_mod(input.emotion.intensity),
        );

        let actual = result.updates[0].new_fidelity.score;
        assert!(
            (actual - expected).abs() < 1e-9,
            "tick should apply exactly the requested duration: actual={actual}, expected={expected}"
        );
    }

    // ── Test 3: Emotional modulation slows decay ─────────────────────────

    #[test]
    fn emotional_modulation_slows_decay() {
        let engine = PowerLawDecayEngine::with_defaults();
        let one_hour_ago = Utc::now() - Duration::hours(1);

        let neutral = make_input(default_fidelity(), default_emotion(), one_hour_ago);

        let mut emotional_vec = default_emotion();
        emotional_vec.intensity = 0.8;
        let emotional = make_input(default_fidelity(), emotional_vec, one_hour_ago);

        let r_neutral = engine.compute_tick(&[neutral], 3600.0);
        let r_emotional = engine.compute_tick(&[emotional], 3600.0);

        let f_neutral = r_neutral.updates[0].new_fidelity.score;
        let f_emotional = r_emotional.updates[0].new_fidelity.score;

        assert!(
            f_emotional > f_neutral,
            "Emotional memory should decay slower: {f_emotional} > {f_neutral}"
        );
    }

    // ── Test 4: Stability increases after retrieval boost ─────────────────

    #[test]
    fn stability_increases_after_retrieval() {
        let engine = PowerLawDecayEngine::with_defaults();
        let s_old = 1.0;
        let s_new = engine.boost_stability(s_old);
        assert!(
            s_new > s_old,
            "Stability should increase: {s_new} > {s_old}"
        );
    }

    #[test]
    fn stability_boost_is_consistent_with_formula() {
        let engine = PowerLawDecayEngine::with_defaults();
        let s_old = 5.0;
        let s_new = engine.boost_stability(s_old);
        let expected = 5.0 * (1.0 + 1.5 * 5.0_f64.powf(-0.2));
        assert!(
            (s_new - expected).abs() < 1e-9,
            "s_new={s_new}, expected={expected}"
        );
    }

    // ── Test 5: Noise accumulates, more when F is lower ──────────────────

    #[test]
    fn noise_increases_over_time() {
        let engine = PowerLawDecayEngine::with_defaults();
        let one_hour_ago = Utc::now() - Duration::hours(1);
        let input = make_input(default_fidelity(), default_emotion(), one_hour_ago);

        let result = engine.compute_tick(&[input], 3600.0);
        let n = result.updates[0].new_fidelity.noise_level;
        assert!(n > 0.0, "Noise should increase after 1 hour, got {n}");
    }

    #[test]
    fn noise_higher_with_lower_fidelity() {
        let engine = PowerLawDecayEngine::with_defaults();
        // Use a short elapsed time and high stability so that noise does not
        // saturate to 1.0 for both cases (which would mask the difference).
        let recently = Utc::now() - Duration::seconds(60);

        // High initial fidelity with high stability (keeps F close to 1.0).
        let mut high_fid = default_fidelity();
        high_fid.stability = 100_000.0;
        let high_f = make_input(high_fid, default_emotion(), recently);

        // Low initial fidelity with the same stability.
        let mut low_fid = default_fidelity();
        low_fid.score = 0.1;
        low_fid.stability = 100_000.0;
        let low_f = make_input(low_fid, default_emotion(), recently);

        let r_high = engine.compute_tick(&[high_f], 60.0);
        let r_low = engine.compute_tick(&[low_f], 60.0);

        let n_high = r_high.updates[0].new_fidelity.noise_level;
        let n_low = r_low.updates[0].new_fidelity.noise_level;

        // Lower fidelity records should accumulate more noise because (1 - F) is larger.
        assert!(
            n_low > n_high,
            "Lower fidelity should mean more noise: {n_low} > {n_high}"
        );
    }

    // ── Test 6: Boundary clamping ────────────────────────────────────────

    #[test]
    fn fidelity_clamped_to_unit_interval() {
        let engine = PowerLawDecayEngine::with_defaults();

        // Very long time - fidelity should not go negative.
        let very_old = Utc::now() - Duration::days(365 * 100);
        let input = make_input(default_fidelity(), default_emotion(), very_old);
        let result = engine.compute_tick(&[input], 1.0);
        let f = result.updates[0].new_fidelity.score;
        assert!(f >= 0.0, "Fidelity must not be negative, got {f}");
        assert!(f <= 1.0, "Fidelity must not exceed 1.0, got {f}");
    }

    #[test]
    fn noise_clamped_to_unit_interval() {
        let engine = PowerLawDecayEngine::with_defaults();

        let very_old = Utc::now() - Duration::days(365 * 100);
        let mut fid = default_fidelity();
        fid.noise_level = 0.99;
        let input = make_input(fid, default_emotion(), very_old);
        let result = engine.compute_tick(&[input], 1.0);
        let n = result.updates[0].new_fidelity.noise_level;
        assert!(n >= 0.0, "Noise must not be negative, got {n}");
        assert!(n <= 1.0, "Noise must not exceed 1.0, got {n}");
    }

    // ── Test 7: Prune threshold ──────────────────────────────────────────

    #[test]
    fn records_below_prune_threshold_flagged() {
        let engine = PowerLawDecayEngine::with_defaults();

        let very_old = Utc::now() - Duration::days(365 * 100);
        let input = make_input(default_fidelity(), default_emotion(), very_old);
        let result = engine.compute_tick(&[input], 1.0);

        let f = result.updates[0].new_fidelity.score;
        let pruned = result.updates[0].should_prune;
        assert!(
            f < 0.01,
            "Very old record should have fidelity < 0.01, got {f}"
        );
        assert!(pruned, "Should be flagged for pruning");
        assert_eq!(result.records_pruned, 1);
        assert_eq!(result.records_below_threshold, 1);
    }

    #[test]
    fn fresh_records_not_flagged_for_pruning() {
        let engine = PowerLawDecayEngine::with_defaults();
        let now = Utc::now();
        let input = make_input(default_fidelity(), default_emotion(), now);
        let result = engine.compute_tick(&[input], 0.0);

        assert!(
            !result.updates[0].should_prune,
            "Fresh record should not be pruned"
        );
        assert_eq!(result.records_pruned, 0);
    }

    // ── Test 8: Batch processing with 10000 records ──────────────────────

    #[test]
    fn batch_processing_10000_records() {
        let engine = PowerLawDecayEngine::with_defaults();
        let base_time = Utc::now() - Duration::hours(1);

        let records: Vec<DecayInput> = (0..10_000)
            .map(|i| {
                let mut fid = default_fidelity();
                // Spread stability values for variety.
                fid.stability = 1.0 + (i as f64);
                let offset = Duration::seconds(i as i64);
                make_input(fid, default_emotion(), base_time + offset)
            })
            .collect();

        let result = engine.compute_tick(&records, 3600.0);

        assert_eq!(
            result.updates.len(),
            10_000,
            "Should process all 10000 records"
        );
        assert_eq!(result.records_updated, 10_000);

        // Verify all outputs are within valid ranges.
        for output in &result.updates {
            assert!(
                (0.0..=1.0).contains(&output.new_fidelity.score),
                "Fidelity out of range: {}",
                output.new_fidelity.score
            );
            assert!(
                (0.0..=1.0).contains(&output.new_fidelity.noise_level),
                "Noise out of range: {}",
                output.new_fidelity.noise_level
            );
        }

        // Records with higher stability should have higher fidelity.
        let first = result.updates[0].new_fidelity.score;
        let last = result.updates[9999].new_fidelity.score;
        assert!(
            last > first,
            "Higher stability should yield higher fidelity: last={last} > first={first}"
        );
    }

    // ── Test 9: Zero tick duration → no change ───────────────────────────

    #[test]
    fn zero_tick_duration_no_change() {
        let engine = PowerLawDecayEngine::with_defaults();
        let now = Utc::now();
        let input = make_input(default_fidelity(), default_emotion(), now);

        let result = engine.compute_tick(&[input], 0.0);
        let f = result.updates[0].new_fidelity.score;
        // The key insight: with last_accessed_at = now, elapsed time is ~0.
        assert!(
            f > 0.99,
            "Zero elapsed time should yield ~unchanged fidelity, got {f}"
        );
    }

    // ── Additional edge case tests ───────────────────────────────────────

    #[test]
    fn empty_input_returns_empty_result() {
        let engine = PowerLawDecayEngine::with_defaults();
        let result = engine.compute_tick(&[], 60.0);
        assert!(result.updates.is_empty());
        assert_eq!(result.records_updated, 0);
        assert_eq!(result.records_below_threshold, 0);
        assert_eq!(result.records_pruned, 0);
    }

    #[test]
    fn custom_params_respected() {
        let params = DecayParams {
            decay_exponent: 0.5,
            retrieval_boost: 2.0,
            interference_rate: 0.05,
            prune_threshold: 0.1,
            initial_fidelity: 0.8,
        };
        let engine = PowerLawDecayEngine::new(params);
        assert!((engine.params().decay_exponent - 0.5).abs() < 1e-9);
        assert!((engine.params().prune_threshold - 0.1).abs() < 1e-9);
    }

    #[test]
    fn higher_prune_threshold_prunes_more() {
        let aggressive_params = DecayParams {
            prune_threshold: 0.5,
            ..DecayParams::default()
        };
        let engine = PowerLawDecayEngine::new(aggressive_params);
        let one_hour_ago = Utc::now() - Duration::hours(1);
        let input = make_input(default_fidelity(), default_emotion(), one_hour_ago);

        let result = engine.compute_tick(&[input], 3600.0);
        let f = result.updates[0].new_fidelity.score;

        // With default stability=1.0, after 1h fidelity is very low.
        // An aggressive threshold of 0.5 should flag it.
        assert!(
            result.updates[0].should_prune,
            "With prune_threshold=0.5, low-fidelity record should be pruned (f={f})"
        );
    }

    #[test]
    fn per_record_decay_rate_used() {
        let engine = PowerLawDecayEngine::with_defaults();
        let one_hour_ago = Utc::now() - Duration::hours(1);

        // Record with low decay exponent (slower decay).
        let mut slow_fid = default_fidelity();
        slow_fid.decay_rate = 0.1;
        slow_fid.stability = 100.0;
        let slow = make_input(slow_fid, default_emotion(), one_hour_ago);

        // Record with high decay exponent (faster decay).
        let mut fast_fid = default_fidelity();
        fast_fid.decay_rate = 0.8;
        fast_fid.stability = 100.0;
        let fast = make_input(fast_fid, default_emotion(), one_hour_ago);

        let r_slow = engine.compute_tick(&[slow], 3600.0);
        let r_fast = engine.compute_tick(&[fast], 3600.0);

        let f_slow = r_slow.updates[0].new_fidelity.score;
        let f_fast = r_fast.updates[0].new_fidelity.score;

        assert!(
            f_slow > f_fast,
            "Lower decay_rate should mean slower decay: {f_slow} > {f_fast}"
        );
    }

    #[test]
    fn emotional_anchor_stored_in_output() {
        let engine = PowerLawDecayEngine::with_defaults();
        let one_hour_ago = Utc::now() - Duration::hours(1);

        let mut emotional_vec = default_emotion();
        emotional_vec.intensity = 0.6;
        let input = make_input(default_fidelity(), emotional_vec, one_hour_ago);

        let result = engine.compute_tick(&[input], 3600.0);
        let e_anchor = result.updates[0].new_fidelity.emotional_anchor;

        // E_mod = 1.0 + 0.6 * 0.5 = 1.3
        assert!(
            (e_anchor - 1.3).abs() < 1e-9,
            "emotional_anchor should be E_mod=1.3, got {e_anchor}"
        );
    }
}
