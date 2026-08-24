"""Run-time / signal-to-noise statistics for HFS peak measurements.

Date:    2026-06-02
Version: 1.0.0
Author:  Arda Kayaalp <arda.kayaalp@kuleuven.be>

Computes the on-peak integration time needed to reach a required
statistical significance, the resulting total scanning time (steps,
sweeps, overhead), and the per-peak timing estimate for the strongest
hyperfine peaks of an isotope. This is the core run-time estimator of
the estimation tab.

Depends on: standard library and third-party packages only (math).
"""
import math


def signal_rate_for_peak(yield_cps, efficiency, peak_relative_intensity):
    """Signal rate [Hz] for a single HFS peak."""
    return yield_cps * efficiency * peak_relative_intensity


def time_for_sigma(signal_rate_Hz, background_rate_Hz, required_sigma):
    """On-peak integration time [s] needed to reach required_sigma significance.

    Formula: t = sigma^2 * (S + B) / S^2
    """
    if signal_rate_Hz <= 0:
        return float("inf")
    return (required_sigma**2 * (signal_rate_Hz + background_rate_Hz)
            / signal_rate_Hz**2)


def scanning_time(t_on_peak_s, voltage_range_V, step_size_V, dwell_time_ms):
    """Compute total scanning time including overhead.

    Parameters
    ----------
    t_on_peak_s : float – required on-peak integration time [s]
    voltage_range_V : float – total voltage scan range [V]
    step_size_V : float – voltage step size [V]
    dwell_time_ms : float – dwell time per step [ms]

    Returns
    -------
    dict with n_steps, time_per_sweep_s, n_sweeps, total_time_s, effective_on_peak_s
    """
    dwell_s = dwell_time_ms / 1000.0
    n_steps = max(1, int(round(voltage_range_V / step_size_V)))
    time_per_sweep_s = n_steps * dwell_s
    if dwell_s <= 0:
        return dict(n_steps=n_steps, time_per_sweep_s=0, n_sweeps=0,
                    total_time_s=0, effective_on_peak_s=0)
    n_sweeps = max(1, math.ceil(t_on_peak_s / dwell_s))
    total_time_s = n_sweeps * time_per_sweep_s
    effective_on_peak_s = n_sweeps * dwell_s
    return dict(
        n_steps=n_steps,
        time_per_sweep_s=time_per_sweep_s,
        n_sweeps=n_sweeps,
        total_time_s=total_time_s,
        effective_on_peak_s=effective_on_peak_s,
    )


def estimate_isotope_timing(
    offsets_MHz,
    intensities,
    n_peaks,
    yield_cps,
    efficiency,
    bg_rate,
    sigma,
    scan_params,
    uniform_timing=False,
):
    """Estimate timing for the strongest n_peaks of an isotope.

    Parameters
    ----------
    offsets_MHz : array – HFS offsets [MHz]
    intensities : array – normalised intensities (sum=1)
    n_peaks : int – number of strongest peaks to measure
    yield_cps : float – production yield [ions/s]
    efficiency : float – spectroscopic efficiency (0..1)
    bg_rate : float – background rate [Hz]
    sigma : float – required statistical significance
    scan_params : dict with voltage_range_V, step_size_V, dwell_time_ms
    uniform_timing : bool – if True, all peaks use the strongest peak's timing

    Returns
    -------
    list of dicts (sorted strongest-first), each containing:
        offset_MHz, intensity, signal_rate_Hz, t_on_peak_s, scan_result (dict)
    """
    order = sorted(range(len(intensities)), key=lambda i: intensities[i], reverse=True)
    n = min(n_peaks, len(order))

    # Pre-compute reference timing assuming full yield (intensity=1) when uniform
    ref_t = None
    ref_scan = None
    if uniform_timing and n > 0:
        ref_sig = signal_rate_for_peak(yield_cps, efficiency, 1.0)
        ref_t = time_for_sigma(ref_sig, bg_rate, sigma)
        ref_scan = scanning_time(
            ref_t,
            scan_params["voltage_range_V"],
            scan_params["step_size_V"],
            scan_params["dwell_time_ms"],
        )

    results = []
    for idx in order[:n]:
        sig = signal_rate_for_peak(yield_cps, efficiency, intensities[idx])
        if uniform_timing:
            t_peak = ref_t
            scan = ref_scan
        else:
            t_peak = time_for_sigma(sig, bg_rate, sigma)
            scan = scanning_time(
                t_peak,
                scan_params["voltage_range_V"],
                scan_params["step_size_V"],
                scan_params["dwell_time_ms"],
            )
        results.append(dict(
            offset_MHz=offsets_MHz[idx],
            intensity=intensities[idx],
            signal_rate_Hz=sig,
            t_on_peak_s=t_peak,
            scan_result=scan,
        ))
    return results
