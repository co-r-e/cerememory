//! Pure mathematical functions for the power-law decay model (ADR-005).
//!
//! All functions are stateless and side-effect-free, making them easy to test
//! and safe for parallel execution via rayon.

/// Compute decayed fidelity using the modified power-law formula.
///
/// ```text
/// F(t) = F_0 * (1 + t/S)^(-d) * E_mod
/// ```
///
/// Returns a value clamped to [0.0, 1.0].
///
/// # Arguments
/// * `f0` - initial fidelity score (0.0..=1.0)
/// * `t_secs` - elapsed time in seconds since last access
/// * `stability` - stability constant S (increases with retrieval)
/// * `decay_exponent` - exponent d (default: 0.3)
/// * `emotion_mod` - emotional modulation factor E_mod (>= 1.0 for emotional memories)
#[inline]
pub fn compute_fidelity(
    f0: f64,
    t_secs: f64,
    stability: f64,
    decay_exponent: f64,
    emotion_mod: f64,
) -> f64 {
    debug_assert!(stability > 0.0, "stability must be positive");

    if t_secs <= 0.0 {
        // No time has passed; fidelity unchanged (but still apply emotion_mod clamping).
        return (f0 * emotion_mod).clamp(0.0, 1.0);
    }

    // (1 + t/S)^(-d)
    let temporal_decay = (1.0 + t_secs / stability).powf(-decay_exponent);

    (f0 * temporal_decay * emotion_mod).clamp(0.0, 1.0)
}

/// Compute accumulated noise level.
///
/// ```text
/// N(t) = N_0 + interference_rate * sqrt(t) * (1 - F(t))
/// ```
///
/// Returns a value clamped to [0.0, 1.0].
///
/// # Arguments
/// * `n0` - initial noise level (0.0..=1.0)
/// * `t_secs` - elapsed time in seconds
/// * `fidelity` - current fidelity F(t) after decay
/// * `interference_rate` - rate constant (default: 0.1)
#[inline]
pub fn compute_noise(n0: f64, t_secs: f64, fidelity: f64, interference_rate: f64) -> f64 {
    if t_secs <= 0.0 {
        return n0.clamp(0.0, 1.0);
    }

    let noise_increment = interference_rate * t_secs.sqrt() * (1.0 - fidelity);

    (n0 + noise_increment).clamp(0.0, 1.0)
}

/// Compute the new stability constant after a retrieval/reinforcement event.
///
/// ```text
/// S_new = S_old * (1 + retrieval_boost * S_old^(-0.2))
/// ```
///
/// # Arguments
/// * `s_old` - current stability constant
/// * `retrieval_boost` - boost constant (default: 1.5)
#[inline]
pub fn compute_stability_boost(s_old: f64, retrieval_boost: f64) -> f64 {
    debug_assert!(s_old > 0.0, "stability must be positive");

    s_old * (1.0 + retrieval_boost * s_old.powf(-0.2))
}

/// Compute the emotional modulation factor from emotion intensity.
///
/// ```text
/// E_mod = 1.0 + emotion_intensity * 0.5
/// ```
///
/// Emotional memories decay more slowly because E_mod > 1.0 acts as a
/// scaling factor that partially counteracts temporal decay.
#[inline]
pub fn compute_emotion_mod(emotion_intensity: f64) -> f64 {
    1.0 + emotion_intensity * 0.5
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f64 = 1e-9;

    // ── Fidelity ──────────────────────────────────────────────────────────

    #[test]
    fn fidelity_no_time_elapsed() {
        let f = compute_fidelity(1.0, 0.0, 1.0, 0.3, 1.0);
        assert!((f - 1.0).abs() < EPSILON, "F(0) should be 1.0, got {f}");
    }

    #[test]
    fn fidelity_decreases_over_time() {
        let f0 = 1.0;
        let f_after_1h = compute_fidelity(f0, 3600.0, 1.0, 0.3, 1.0);
        assert!(
            f_after_1h < f0,
            "Fidelity should decrease over time: {f_after_1h} < {f0}"
        );
        assert!(
            f_after_1h > 0.0,
            "Fidelity should still be positive: {f_after_1h}"
        );
    }

    #[test]
    fn fidelity_decreases_meaningfully_after_1h() {
        let f = compute_fidelity(1.0, 3600.0, 1.0, 0.3, 1.0);
        // With S=1, d=0.3, (1 + 3600)^(-0.3) is very small.
        assert!(f < 0.1, "After 1h with S=1, F should be very low, got {f}");
    }

    #[test]
    fn fidelity_higher_stability_slows_decay() {
        let f_low_s = compute_fidelity(1.0, 3600.0, 1.0, 0.3, 1.0);
        let f_high_s = compute_fidelity(1.0, 3600.0, 10000.0, 0.3, 1.0);
        assert!(
            f_high_s > f_low_s,
            "Higher stability should slow decay: {f_high_s} > {f_low_s}"
        );
    }

    #[test]
    fn fidelity_emotional_modulation_slows_decay() {
        let f_neutral = compute_fidelity(1.0, 3600.0, 100.0, 0.3, 1.0);
        let f_emotional = compute_fidelity(1.0, 3600.0, 100.0, 0.3, 1.5);
        assert!(
            f_emotional > f_neutral,
            "Emotional modulation should slow decay: {f_emotional} > {f_neutral}"
        );
    }

    #[test]
    fn fidelity_clamped_to_unit_interval() {
        // E_mod can push above 1.0 at t=0 — should be clamped.
        let f = compute_fidelity(1.0, 0.0, 1.0, 0.3, 2.0);
        assert!(f <= 1.0, "Fidelity must be clamped to 1.0, got {f}");

        // Very large time should not go negative.
        let f_large_t = compute_fidelity(1.0, 1e12, 1.0, 0.3, 1.0);
        assert!(
            f_large_t >= 0.0,
            "Fidelity must not go negative, got {f_large_t}"
        );
    }

    #[test]
    fn fidelity_zero_initial_stays_zero() {
        let f = compute_fidelity(0.0, 3600.0, 100.0, 0.3, 1.0);
        assert!((f - 0.0).abs() < EPSILON, "F(0) * anything = 0, got {f}");
    }

    // ── Noise ─────────────────────────────────────────────────────────────

    #[test]
    fn noise_no_time_elapsed() {
        let n = compute_noise(0.0, 0.0, 1.0, 0.1);
        assert!((n - 0.0).abs() < EPSILON, "N(0) should be 0.0, got {n}");
    }

    #[test]
    fn noise_increases_over_time() {
        let n0 = 0.0;
        let n = compute_noise(n0, 3600.0, 0.5, 0.1);
        assert!(n > n0, "Noise should increase over time: {n} > {n0}");
    }

    #[test]
    fn noise_higher_when_fidelity_lower() {
        let n_high_f = compute_noise(0.0, 3600.0, 0.9, 0.1);
        let n_low_f = compute_noise(0.0, 3600.0, 0.1, 0.1);
        assert!(
            n_low_f > n_high_f,
            "Lower fidelity should produce more noise: {n_low_f} > {n_high_f}"
        );
    }

    #[test]
    fn noise_no_increase_when_fidelity_is_one() {
        let n = compute_noise(0.0, 3600.0, 1.0, 0.1);
        assert!(
            (n - 0.0).abs() < EPSILON,
            "No noise should accumulate when F=1.0, got {n}"
        );
    }

    #[test]
    fn noise_clamped_to_unit_interval() {
        let n = compute_noise(0.9, 1e12, 0.0, 1.0);
        assert!(n <= 1.0, "Noise must be clamped to 1.0, got {n}");
    }

    // ── Stability ─────────────────────────────────────────────────────────

    #[test]
    fn stability_increases_on_retrieval() {
        let s_old = 1.0;
        let s_new = compute_stability_boost(s_old, 1.5);
        assert!(
            s_new > s_old,
            "Stability should increase after retrieval: {s_new} > {s_old}"
        );
    }

    #[test]
    fn stability_boost_diminishes_with_high_stability() {
        // The multiplicative boost factor should be smaller for higher S values.
        let boost_low = compute_stability_boost(1.0, 1.5) / 1.0;
        let boost_high = compute_stability_boost(100.0, 1.5) / 100.0;
        assert!(
            boost_low > boost_high,
            "Boost ratio should diminish: {boost_low} > {boost_high}"
        );
    }

    // ── Emotion mod ───────────────────────────────────────────────────────

    #[test]
    fn emotion_mod_neutral() {
        let e = compute_emotion_mod(0.0);
        assert!((e - 1.0).abs() < EPSILON);
    }

    #[test]
    fn emotion_mod_high_intensity() {
        let e = compute_emotion_mod(1.0);
        assert!((e - 1.5).abs() < EPSILON);
    }
}
