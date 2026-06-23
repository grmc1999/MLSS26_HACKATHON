"""ILI forecasting data: CDC ILINet (US, pretrain/source domain) + WHO FluID (fine-tune/target countries).

Both APIs are public, unauthenticated HTTP/OData endpoints. Fetches use stdlib
urllib only (no new dependency) and are cached to disk under data_cache/.

The forecasting task is 5 past epiweeks -> 10 future epiweeks of an %ILI signal.
Both domains' value column is already a percentage-of-population-style rate
(CDC's `wili`, WHO's derived `pct_ili`), so the SIR physical model below is kept
in percentage space with a fixed N=100 rather than estimating real populations.
"""
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

CACHE_DIR = Path(__file__).resolve().parent / "data_cache"
CACHE_DIR.mkdir(exist_ok=True)

CDC_BASE_URL = "https://api.delphi.cmu.edu/epidata/fluview/"
WHO_BASE_URL = "https://xmart-api-public.who.int/FLUMART/VIW_FID_EPI"

# Fine-tuning/target domains: genuinely different countries from the US pretrain
# domain, picked for data density + hemisphere/climate diversity.
TARGET_COUNTRIES = ["FRA", "MEX", "AUS", "ZAF"]

INPUT_STEPS = 5
FORECAST_STEPS = 10
SIR_N = 100.0  # both value columns are %-of-population scale


def _http_get_json(url, retries=3, timeout=30):
    last_err = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            last_err = e
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"Failed to fetch {url}: {last_err}")


def epiweek_to_season(epiweek):
    """Flu season = MMWR week 40 of year Y through week 39 of year Y+1."""
    year, week = divmod(int(epiweek), 100)
    if week >= 40:
        return f"{year}-{year + 1}"
    return f"{year - 1}-{year}"


def fetch_cdc_ilinet(region="nat", start_epiweek=199740, end_epiweek=202625, force_refresh=False):
    cache_path = CACHE_DIR / f"cdc_{region}_{start_epiweek}_{end_epiweek}.json"
    if cache_path.exists() and not force_refresh:
        raw = json.loads(cache_path.read_text())
    else:
        url = CDC_BASE_URL + "?" + urllib.parse.urlencode(
            {"regions": region, "epiweeks": f"{start_epiweek}-{end_epiweek}"}
        )
        raw = _http_get_json(url)
        if raw.get("result") != 1:
            raise RuntimeError(f"CDC fluview API error: {raw.get('message')}")
        cache_path.write_text(json.dumps(raw))
    df = pd.DataFrame(raw["epidata"])[["epiweek", "wili"]]
    return df.dropna(subset=["wili"]).sort_values("epiweek").reset_index(drop=True)


def fetch_who_fluid(country_code, force_refresh=False):
    cache_path = CACHE_DIR / f"who_{country_code}.json"
    if cache_path.exists() and not force_refresh:
        rows = json.loads(cache_path.read_text())
    else:
        rows = []
        url = WHO_BASE_URL + "?" + urllib.parse.urlencode(
            {"$filter": f"COUNTRY_CODE eq '{country_code}'", "$top": "20000"}
        )
        while url:
            page = _http_get_json(url)
            rows.extend(page.get("value", []))
            url = page.get("@odata.nextLink")
        cache_path.write_text(json.dumps(rows))

    if not rows:
        raise RuntimeError(f"No WHO FluID data returned for country {country_code}")

    # WHO FluID's ILI_OUTPATIENTS/ILI_POP_COV denominators are unreliable across
    # most countries (verified: only ~France reports them consistently), so a
    # true %ILI rate can't be computed uniformly. Instead, scale raw reported
    # case counts by the series' own 99th percentile into a bounded [0, ~100+]
    # relative-intensity proxy, comparable in spirit to CDC's %ILI for SIR purposes.
    df = pd.DataFrame(rows)
    grouped = df.groupby("MMWRYW").agg(
        case=("ILI_CASE", "sum"), n_reports=("ILI_CASE", "count")
    ).reset_index()
    grouped = grouped[grouped["n_reports"] > 0].copy()
    scale = max(float(grouped["case"].quantile(0.99)), 1.0)
    grouped["pct_ili"] = 100.0 * grouped["case"] / scale
    grouped = grouped.rename(columns={"MMWRYW": "epiweek"})
    return grouped[["epiweek", "pct_ili"]].sort_values("epiweek").reset_index(drop=True)


def _epiweek_ordinal(epiweek):
    """Approximate week-ordinal (52 weeks/year) used only to detect gaps between
    consecutive rows of an already-sorted series; off by one near rare 53-week
    years, which only causes a few extra windows near a year boundary to be
    conservatively rejected, never accepted across a real gap."""
    year, week = divmod(int(epiweek), 100)
    return year * 52 + (week - 1)


def _build_series(df, value_col):
    df = df.copy()
    df["season"] = df["epiweek"].apply(epiweek_to_season)
    S = np.zeros(len(df), dtype=float)
    cum, prev_season = 0.0, None
    values = df[value_col].to_numpy(dtype=float)
    seasons = df["season"].to_numpy()
    for i, (v, season) in enumerate(zip(values, seasons)):
        if season != prev_season:
            cum = 0.0
            prev_season = season
        cum += max(v, 0.0)
        S[i] = np.clip(SIR_N - cum, 0.0, SIR_N)
    df["S"] = S
    return df


def make_windows(series_df, value_col, input_steps=INPUT_STEPS, forecast_steps=FORECAST_STEPS, stride=1):
    epiweeks = series_df["epiweek"].to_numpy()
    values = series_df[value_col].to_numpy(dtype=float)
    S = series_df["S"].to_numpy(dtype=float)
    value_by_epiweek = dict(zip(epiweeks.tolist(), values.tolist()))
    total = input_steps + forecast_steps
    windows = []
    for start in range(0, len(series_df) - total + 1, stride):
        idx = np.arange(start, start + total)
        ords = [_epiweek_ordinal(e) for e in epiweeks[idx]]
        if any(b - a != 1 for a, b in zip(ords[:-1], ords[1:])):
            continue
        x, y, s_win = values[idx[:input_steps]], values[idx[input_steps:]], S[idx]
        if np.isnan(x).any() or np.isnan(y).any():
            continue
        y_epiweeks = epiweeks[idx[input_steps:]]
        # seasonal-naive baseline target: same calendar epiweek one year prior
        # (epiweek - 100), falling back to the last observed input value if missing
        naive = np.array([value_by_epiweek.get(int(ew) - 100, x[-1]) for ew in y_epiweeks])
        windows.append({
            "x": x,
            "y": y,
            "S": s_win,
            "naive": naive,
            "season": epiweek_to_season(int(epiweeks[idx[input_steps]])),
        })
    return windows


def assign_season_splits(windows, train_frac=0.7, val_frac=0.15):
    seasons = sorted({w["season"] for w in windows})
    n = len(seasons)
    n_train = max(1, round(n * train_frac))
    n_val = max(1, round(n * val_frac))
    train_seasons = set(seasons[:n_train])
    val_seasons = set(seasons[n_train:n_train + n_val])
    test_seasons = set(seasons[n_train + n_val:]) or set(seasons[-1:])
    train = [w for w in windows if w["season"] in train_seasons]
    val = [w for w in windows if w["season"] in val_seasons]
    test = [w for w in windows if w["season"] in test_seasons]
    return train, val, test


def fit_sir_params(I, S, N=SIR_N, dt=1.0):
    """Closed-form OLS fit of the discrete SIR Euler residual's (beta, gamma),
    since dI/dt = beta*S*I/N - gamma*I is linear in (beta, gamma)."""
    I, S = np.asarray(I, dtype=float), np.asarray(S, dtype=float)
    dI = (I[1:] - I[:-1]) / dt
    X = np.stack([S[:-1] * I[:-1] / N, -I[:-1]], axis=1)
    mask = np.isfinite(dI) & np.all(np.isfinite(X), axis=1)
    X, dI = X[mask], dI[mask]
    if len(dI) < 5:
        return 0.5, 0.3
    coef, *_ = np.linalg.lstsq(X, dI, rcond=None)
    # clip to physically valid (non-negative) transmission/recovery rates;
    # an unconstrained OLS fit can otherwise return a sign-flipped, non-physical pair.
    return max(float(coef[0]), 1e-3), max(float(coef[1]), 1e-3)


def _normalize_windows(train_w, val_w, test_w):
    all_vals = np.concatenate([np.concatenate([w["x"], w["y"]]) for w in train_w])
    mean, std = float(all_vals.mean()), float(all_vals.std() + 1e-6)
    for split in (train_w, val_w, test_w):
        for w in split:
            w["x_norm"] = (w["x"] - mean) / std
            w["y_norm"] = (w["y"] - mean) / std
            w["naive_norm"] = (w["naive"] - mean) / std
    return {"mean": mean, "std": std}


class FluWindowDataset(Dataset):
    def __init__(self, windows):
        self.windows = windows

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        w = self.windows[idx]
        x = torch.tensor(w["x_norm"], dtype=torch.float32).unsqueeze(-1)
        y = torch.tensor(w["y_norm"], dtype=torch.float32).unsqueeze(-1)
        S = torch.tensor(w["S"], dtype=torch.float32)
        naive = torch.tensor(w["naive_norm"], dtype=torch.float32).unsqueeze(-1)
        return x, y, S, SIR_N, naive


def _windows_to_loaders(train_w, val_w, test_w, batch_size):
    stats = _normalize_windows(train_w, val_w, test_w)
    train_loader = DataLoader(FluWindowDataset(train_w), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(FluWindowDataset(val_w), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(FluWindowDataset(test_w), batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader, stats


def get_pretrain_series(region="nat"):
    """US source-domain series (+S trajectory), exposed for SIR parameter fitting."""
    return _build_series(fetch_cdc_ilinet(region=region), "wili"), "wili"


def get_finetune_series(country_code):
    """Target-country series (+S trajectory), exposed for SIR parameter fitting."""
    return _build_series(fetch_who_fluid(country_code), "pct_ili"), "pct_ili"


def load_pretrain_data(batch_size=64, input_steps=INPUT_STEPS, forecast_steps=FORECAST_STEPS, region="nat"):
    df, _ = get_pretrain_series(region=region)
    windows = make_windows(df, "wili", input_steps, forecast_steps)
    train_w, val_w, test_w = assign_season_splits(windows)
    train_loader, val_loader, test_loader, _ = _windows_to_loaders(train_w, val_w, test_w, batch_size)
    return train_loader, val_loader, test_loader


def load_finetune_data(country_code, n_seasons, batch_size=16, input_steps=INPUT_STEPS, forecast_steps=FORECAST_STEPS):
    df, _ = get_finetune_series(country_code)
    windows = make_windows(df, "pct_ili", input_steps, forecast_steps)
    train_w, val_w, test_w = assign_season_splits(windows)
    train_seasons = sorted({w["season"] for w in train_w})
    keep = set(train_seasons[-n_seasons:]) if n_seasons < len(train_seasons) else set(train_seasons)
    train_w = [w for w in train_w if w["season"] in keep]
    train_loader, val_loader, test_loader, stats = _windows_to_loaders(train_w, val_w, test_w, batch_size)
    return train_loader, val_loader, test_loader, stats
