"""NIST ASD access: fetch, parse, cache, and derived atomic quantities.

Date:    2026-07-25
Version: 1.0.0
Author:  Arda Kayaalp <arda.kayaalp@kuleuven.be>

Qt-free data layer for the NIST ASD browser. Talks to the ASD CGI
endpoints with stdlib urllib using parameter sets captured from real
browser form submissions on 2026-07-25 (NIST renamed several fields
since the standalone prototypes — e.g. ``forbidden_out`` became
``forbid_out``, ``en_unit=0`` is now cm⁻¹ — and the CGI rejects
requests that carry unchecked-checkbox parameters, so the sets below
must be reproduced verbatim, only ever substituting the spectrum).

Wavelength policy: lines are fetched with VACUUM wavelengths
(``show_av=2``) as the canonical dataset; air wavelengths are computed
locally with the standard NIST dispersion formula so the air/vacuum
toggle works offline and consistently.

The Type column: ASD leaves Type BLANK for allowed E1 lines and labels
only the forbidden ones (E2, M1, ...). :func:`clean_lines_df`
normalizes blank Type to "E1".

Caching: ``<settings dir>/nist_cache/<Spectrum>_{lines,levels}.csv``,
offline-first — the UI only hits NIST on an explicit fetch/refresh.

Depends on: pandas, numpy; gui.shared_widgets lazily (settings dir).
"""

from __future__ import annotations

import io
import json
import os
import re
import time
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd


class NistFetchError(RuntimeError):
    """Raised when NIST returns an error page instead of data."""


_UA = {"User-Agent": "Mozilla/5.0 (DENIS spectroscopy toolkit)"}

# Captured 2026-07-25 from the live lines form (order preserved).
# Tab-delimited, vacuum wavelengths, energies in cm-1, allowed +
# forbidden lines, conf/term/J, Aki, accuracy, intensities, refs.
_LINES_PARAMS = [
    ("spectra", "{spectrum}"),
    ("output_type", "0"),
    ("low_w", ""),
    ("upp_w", ""),
    ("unit", "1"),
    ("de", "0"),
    ("plot_out", "0"),
    ("I_scale_type", "1"),
    ("format", "3"),
    ("line_out", "0"),
    ("en_unit", "0"),
    ("output", "0"),
    ("bibrefs", "1"),
    ("page_size", "15"),
    ("show_obs_wl", "1"),
    ("show_calc_wl", "1"),
    ("unc_out", "1"),
    ("order_out", "0"),
    ("max_low_enrg", ""),
    ("show_av", "2"),
    ("max_upp_enrg", ""),
    ("tsb_value", "0"),
    ("min_str", ""),
    ("A_out", "0"),
    ("intens_out", "on"),
    ("max_str", ""),
    ("allowed_out", "1"),
    ("forbid_out", "1"),
    ("min_accur", ""),
    ("min_intens", ""),
    ("conf_out", "on"),
    ("term_out", "on"),
    ("enrg_out", "on"),
    ("J_out", "on"),
]

# Captured 2026-07-25 from the live levels form.
_LEVELS_PARAMS = [
    ("de", "0"),
    ("spectrum", "{spectrum}"),
    ("units", "0"),
    ("format", "3"),
    ("output", "0"),
    ("page_size", "15"),
    ("multiplet_ordered", "0"),
    ("conf_out", "on"),
    ("term_out", "on"),
    ("level_out", "on"),
    ("unc_out", "1"),
    ("j_out", "on"),
    ("lande_out", "on"),
    ("perc_out", "on"),
    ("biblio", "on"),
    ("temp", ""),
]

_LINES_URL = "https://physics.nist.gov/cgi-bin/ASD/lines1.pl"
_LEVELS_URL = "https://physics.nist.gov/cgi-bin/ASD/energy1.pl"


def _http_get(url: str, params: list, spectrum: str, timeout: float) -> str:
    qs = urllib.parse.urlencode(
        [(k, v.format(spectrum=spectrum)) for k, v in params])
    req = urllib.request.Request(f"{url}?{qs}", headers=_UA)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="replace")


def _raise_if_error_page(text: str, spectrum: str) -> None:
    if not text.lstrip().startswith("<"):
        return
    body = re.sub(r"<script.*?</script>", " ", text, flags=re.S)
    body = re.sub(r"<[^>]+>", " ", body)
    body = re.sub(r"\s+", " ", body).strip()
    m = re.search(r"Error Message:\s*(.{0,120})", body)
    msg = m.group(1).strip() if m else body[:120]
    raise NistFetchError(
        f"NIST ASD returned an error for '{spectrum}': {msg}")


def parse_tab_delimited(text: str) -> pd.DataFrame:
    """Parse ASD tab-delimited output (quoted cells, trailing tab)."""
    df = pd.read_csv(io.StringIO(text), sep="\t")
    df.columns = [str(c).replace('"', "").strip() for c in df.columns]
    # The trailing tab on every row yields one all-empty unnamed column.
    drop = [c for c in df.columns if c.startswith("Unnamed")]
    return df.drop(columns=drop)


def clean_lines_df(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize a raw lines table: numeric coercion, string columns,
    blank Type → "E1" (ASD labels only forbidden lines). Rows without
    level energies (observed-only lines) are KEPT for browsing; use
    :func:`transitions_only` for physics that needs both levels."""
    df = df.copy()
    for col in ("obs_wl_vac(nm)", "ritz_wl_vac(nm)", "obs_wl_air(nm)",
                "ritz_wl_air(nm)", "Aki(s^-1)", "Ei(cm-1)", "Ek(cm-1)"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ("conf_i", "term_i", "J_i", "conf_k", "term_k", "J_k",
                "Type", "Acc", "intens"):
        if col in df.columns:
            df[col] = (df[col].fillna("").astype(str).str.strip()
                       .replace("nan", ""))
    if "Type" in df.columns:
        df["Type"] = df["Type"].replace("", "E1")
    return df


def transitions_only(lines_df: pd.DataFrame) -> pd.DataFrame:
    """Rows with both level energies known (usable for schemes)."""
    if (lines_df is None or lines_df.empty
            or "Ei(cm-1)" not in lines_df.columns
            or "Ek(cm-1)" not in lines_df.columns):
        return pd.DataFrame(columns=(lines_df.columns
                                     if lines_df is not None
                                     else []))
    return lines_df.dropna(subset=["Ei(cm-1)", "Ek(cm-1)"])


def fetch_lines_from_nist(spectrum: str, timeout: float = 60.0
                          ) -> pd.DataFrame:
    """Live query: all lines of ``spectrum`` (e.g. "Si I")."""
    text = _http_get(_LINES_URL, _LINES_PARAMS, spectrum, timeout)
    _raise_if_error_page(text, spectrum)
    return clean_lines_df(parse_tab_delimited(text))


def clean_levels_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ("Level (cm-1)", "Uncertainty (cm-1)", "Lande"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ("Configuration", "Term", "J", "Leading percentages"):
        if col in df.columns:
            df[col] = (df[col].fillna("").astype(str).str.strip()
                       .replace("nan", ""))
    drop = [c for c in ("Prefix", "Suffix") if c in df.columns]
    df = df.drop(columns=drop)
    return df.dropna(subset=["Level (cm-1)"]).reset_index(drop=True)


def fetch_levels_from_nist(spectrum: str, timeout: float = 60.0
                           ) -> pd.DataFrame:
    """Live query: all energy levels of ``spectrum``."""
    text = _http_get(_LEVELS_URL, _LEVELS_PARAMS, spectrum, timeout)
    _raise_if_error_page(text, spectrum)
    return clean_levels_df(parse_tab_delimited(text))


def derive_levels_from_lines(lines_df: pd.DataFrame) -> pd.DataFrame:
    """Unique levels reconstructed from the line list — the fallback
    when energy1.pl is unreachable (levels that emit no lines are
    invisible to this route)."""
    t = transitions_only(lines_df)
    lower = t[["Ei(cm-1)", "conf_i", "term_i", "J_i"]].rename(columns={
        "Ei(cm-1)": "Level (cm-1)", "conf_i": "Configuration",
        "term_i": "Term", "J_i": "J"})
    upper = t[["Ek(cm-1)", "conf_k", "term_k", "J_k"]].rename(columns={
        "Ek(cm-1)": "Level (cm-1)", "conf_k": "Configuration",
        "term_k": "Term", "J_k": "J"})
    both = pd.concat([lower, upper]).dropna(subset=["Level (cm-1)"])
    return (both.drop_duplicates(subset=["Level (cm-1)"])
            .sort_values("Level (cm-1)").reset_index(drop=True))


# ── Cache ────────────────────────────────────────────────────────────────

def cache_dir() -> str:
    """``<settings dir>/nist_cache`` (created on demand)."""
    from gui.shared_widgets import _DEFAULT_SETTINGS_DIR
    d = os.path.join(_DEFAULT_SETTINGS_DIR, "nist_cache")
    os.makedirs(d, exist_ok=True)
    return d


def _cache_base(spectrum: str, kind: str, directory: str | None) -> str:
    safe = re.sub(r"[^A-Za-z0-9+_-]", "_", spectrum.strip())
    return os.path.join(directory or cache_dir(), f"{safe}_{kind}")


def save_cache(spectrum: str, kind: str, df: pd.DataFrame,
               directory: str | None = None) -> str:
    base = _cache_base(spectrum, kind, directory)
    df.to_csv(base + ".csv", index=False)
    with open(base + ".meta.json", "w", encoding="utf-8") as f:
        json.dump({"spectrum": spectrum, "kind": kind,
                   "rows": int(len(df)), "fetched_at": time.time()}, f)
    return base + ".csv"


def load_cache(spectrum: str, kind: str,
               directory: str | None = None) -> pd.DataFrame | None:
    base = _cache_base(spectrum, kind, directory)
    path = base + ".csv"
    if not os.path.isfile(path):
        return None
    df = pd.read_csv(path)
    return (clean_lines_df(df) if kind == "lines"
            else clean_levels_df(df))


def cache_info(spectrum: str, kind: str,
               directory: str | None = None) -> dict | None:
    base = _cache_base(spectrum, kind, directory)
    meta = base + ".meta.json"
    if not os.path.isfile(meta):
        return None
    try:
        with open(meta, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def get_lines(spectrum: str, refresh: bool = False,
              directory: str | None = None) -> pd.DataFrame:
    """Cache-first lines access; hits NIST only on miss or refresh."""
    if not refresh:
        cached = load_cache(spectrum, "lines", directory)
        if cached is not None:
            return cached
    df = fetch_lines_from_nist(spectrum)
    save_cache(spectrum, "lines", df, directory)
    return df


def get_levels(spectrum: str, refresh: bool = False,
               directory: str | None = None) -> pd.DataFrame:
    """Cache-first levels access with derive-from-lines fallback."""
    if not refresh:
        cached = load_cache(spectrum, "levels", directory)
        if cached is not None:
            return cached
    try:
        df = fetch_levels_from_nist(spectrum)
    except (NistFetchError, OSError):
        lines = get_lines(spectrum, refresh=False, directory=directory)
        df = derive_levels_from_lines(lines)
    save_cache(spectrum, "levels", df, directory)
    return df


# ── Derived physics ──────────────────────────────────────────────────────

def vac_to_air(wl_vac_nm: float) -> float:
    """Standard NIST dispersion conversion (five-figure formula)."""
    if not np.isfinite(wl_vac_nm) or wl_vac_nm <= 0:
        return float("nan")
    s2 = (1e3 / wl_vac_nm) ** 2  # (1/λ in µm⁻¹)²
    n = (1 + 0.0000834254 + 0.02406147 / (130 - s2)
         + 0.00015998 / (38.9 - s2))
    return wl_vac_nm / n


def wavelength_nm(row, medium: str = "vacuum") -> float:
    """Transition wavelength from a lines row (dict or Series).

    Prefers the tabulated Ritz vacuum wavelength; falls back to
    1e7/ΔE. ``medium`` "air" applies the local dispersion conversion.
    """
    get = row.get if hasattr(row, "get") else row.__getitem__
    wl_vac = get("ritz_wl_vac(nm)", None)
    try:
        wl_vac = float(wl_vac)
    except (TypeError, ValueError):
        wl_vac = float("nan")
    if not np.isfinite(wl_vac) or wl_vac <= 0:
        try:
            de = abs(float(get("Ek(cm-1)", np.nan))
                     - float(get("Ei(cm-1)", np.nan)))
        except (TypeError, ValueError):
            return float("nan")
        if not np.isfinite(de) or de == 0:
            return float("nan")
        wl_vac = 1e7 / de
    return vac_to_air(wl_vac) if medium == "air" else wl_vac


def display_wavelength_nm(row, medium: str = "vacuum") -> float:
    """Table-display wavelength: like :func:`wavelength_nm` but falls
    back to the OBSERVED wavelength for observed-only lines (rows
    without level energies), so the browser never shows an empty λ for
    a line NIST does tabulate."""
    wl = wavelength_nm(row, medium)
    if np.isfinite(wl):
        return wl
    get = row.get if hasattr(row, "get") else row.__getitem__
    try:
        obs = float(get("obs_wl_vac(nm)", float("nan")))
    except (TypeError, ValueError):
        return float("nan")
    if not np.isfinite(obs) or obs <= 0:
        return float("nan")
    return vac_to_air(obs) if medium == "air" else obs


def parse_orbital(config: str) -> str | None:
    """Last orbital letter of a configuration ('3s2.3p.4d' → 'd')."""
    if not isinstance(config, str) or config in ("", "nan"):
        return None
    last = config.split(".")[-1]
    m = re.findall(r"\d+([spdfgh])\d*$", last)
    return m[0] if m else None


def _config_to_counts(config_str: str) -> dict:
    if not isinstance(config_str, str) or config_str in ("", "nan"):
        return {}
    cleaned = re.sub(r"\(.*?\)|[ _^]", "", config_str)
    counts: dict = {}
    for part in cleaned.split("."):
        m = re.match(r"(\d+[spdfgh])(\d*)", part)
        if m:
            counts[m.group(1)] = int(m.group(2)) if m.group(2) else 1
    return counts


def orbital_transition(conf_i: str, conf_k: str) -> str | None:
    """Classify the active electron jump: "s->p", "complex", or None
    when either configuration is unparseable."""
    ci, ck = _config_to_counts(conf_i), _config_to_counts(conf_k)
    if not ci or not ck:
        return None
    lost, gained = [], []
    for orb in set(ci) | set(ck):
        a, b = ci.get(orb, 0), ck.get(orb, 0)
        if a > b:
            lost.append(re.sub(r"\d", "", orb))
        elif b > a:
            gained.append(re.sub(r"\d", "", orb))
    if len(lost) == 1 and len(gained) == 1:
        return f"{lost[0]}->{gained[0]}"
    return "complex"


def find_metastable_states(levels_df: pd.DataFrame,
                           lines_df: pd.DataFrame,
                           fast_aki_threshold: float = 1.0e4
                           ) -> pd.DataFrame:
    """Excited levels with NO fast allowed (E1) decay — candidates for
    scheme starting points. A level qualifies when it has decay
    channels but none of them is an E1 line faster than the threshold.
    (Blank ASD Type = E1; normalized by :func:`clean_lines_df`.)"""
    if levels_df.empty or lines_df.empty:
        return pd.DataFrame()
    t = transitions_only(lines_df)
    out = []
    for _, lv in levels_df[levels_df["Level (cm-1)"] > 0].iterrows():
        e = float(lv["Level (cm-1)"])
        decays = t[(t["Ek(cm-1)"] - e).abs() < 0.1]
        if decays.empty:
            continue
        fast_e1 = decays[(decays.get("Type", "E1") == "E1")
                         & (decays["Aki(s^-1)"] > fast_aki_threshold)]
        if fast_e1.empty:
            d = lv.to_dict()
            mx = decays["Aki(s^-1)"].max()
            d["Max Decay Aki"] = float(mx) if pd.notna(mx) else 0.0
            types = sorted(x for x in decays["Type"].dropna().unique()
                           if x)
            d["Decay Types"] = ", ".join(types) if types else "None"
            out.append(d)
    return pd.DataFrame(out)


def find_closest_level(levels_df: pd.DataFrame, target_cm1: float
                       ) -> tuple[float, float]:
    """(closest tabulated level, |difference|); (nan, inf) if empty."""
    if levels_df.empty:
        return float("nan"), float("inf")
    diffs = (levels_df["Level (cm-1)"] - target_cm1).abs()
    idx = diffs.idxmin()
    return float(levels_df.loc[idx, "Level (cm-1)"]), float(diffs.min())
