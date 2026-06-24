"""Fixed evaluation metrics for ILI forecasting (frozen once the autoresearch loop starts).

Loader batches are 5-tuples: (x, y, S, N, naive) -- see env/data.py.
"""
import numpy as np
import torch


def _predict(model, x):
    """Uniform forecast interface: diffusion models expose .sample(x); plain
    regressors are called directly."""
    if hasattr(model, "sample"):
        return model.sample(x)
    return model(x)


def mae(y_true, y_pred):
    return float(torch.mean(torch.abs(y_true - y_pred)))


def per_horizon_mae(y_true, y_pred):
    return torch.mean(torch.abs(y_true - y_pred), dim=(0, 2)).detach().cpu().numpy()


def weighted_horizon_score(per_horizon, weights=None):
    per_horizon = np.asarray(per_horizon)
    if weights is None:
        weights = np.linspace(1.5, 0.5, len(per_horizon))
    return float(np.sum(weights * per_horizon) / np.sum(weights))


def peak_timing_error(y_true, y_pred):
    true_idx = torch.argmax(y_true.squeeze(-1), dim=1)
    pred_idx = torch.argmax(y_pred.squeeze(-1), dim=1)
    return float(torch.mean(torch.abs(true_idx - pred_idx).float()))


def peak_magnitude_error(y_true, y_pred):
    true_peak = torch.amax(y_true.squeeze(-1), dim=1)
    pred_peak = torch.amax(y_pred.squeeze(-1), dim=1)
    return float(torch.mean(torch.abs(true_peak - pred_peak)))


def sir_residual(I_hat, S, N, beta, gamma, dt=1.0):
    """Discrete Euler SIR residual ||I_hat[t+1] - Phi_SIR(I_hat[t], S[t])||,
    averaged over consecutive horizon steps and the batch (Project_definition.md S5.2)."""
    I_t, I_t1 = I_hat[:, :-1], I_hat[:, 1:]
    S_t = S[:, :-1]
    N = N.view(-1, 1)
    phi = I_t + dt * (beta * S_t * I_t / N - gamma * I_t)
    return torch.mean(torch.abs(I_t1 - phi))


def positivity_violation(y_pred):
    return float(torch.mean(torch.clamp(-y_pred, min=0.0)))


def evaluate_full(model, loader, device, beta=0.5, gamma=0.3, stats=None):
    """Runs `model` over `loader`, computing every metric in one pass.
    If `stats` (the {"mean","std"} normalization dict) is given, values are
    denormalized first so MAE etc. are reported in real %ILI-proxy units."""
    model.eval()
    all_y, all_pred, all_S, all_N = [], [], [], []
    with torch.no_grad():
        for x, y, S, N, _naive in loader:
            x, y, S, N = x.to(device), y.to(device), S.to(device), N.to(device)
            y_hat = _predict(model, x)
            all_y.append(y)
            all_pred.append(y_hat)
            all_S.append(S)
            all_N.append(N)
    y_true, y_pred = torch.cat(all_y), torch.cat(all_pred)
    S, N = torch.cat(all_S), torch.cat(all_N)

    if stats is not None:
        y_true = y_true * stats["std"] + stats["mean"]
        y_pred = y_pred * stats["std"] + stats["mean"]

    ph = per_horizon_mae(y_true, y_pred)
    S_forecast = S[:, -y_pred.shape[1]:]
    return {
        "mae": mae(y_true, y_pred),
        "per_horizon_mae": ph.tolist(),
        "weighted_horizon_score": weighted_horizon_score(ph),
        "peak_timing_error": peak_timing_error(y_true, y_pred),
        "peak_magnitude_error": peak_magnitude_error(y_true, y_pred),
        "sir_residual": float(sir_residual(y_pred.squeeze(-1), S_forecast, N, beta, gamma)),
        "positivity_violation": positivity_violation(y_pred),
    }


def evaluate_seasonal_naive(loader, stats=None):
    """The naive baseline needs no model: the loader already carries the
    same-calendar-week-last-season target precomputed in env/data.py."""
    all_y, all_pred = [], []
    for x, y, S, N, naive in loader:
        all_y.append(y)
        all_pred.append(naive)
    y_true, y_pred = torch.cat(all_y), torch.cat(all_pred)
    if stats is not None:
        y_true = y_true * stats["std"] + stats["mean"]
        y_pred = y_pred * stats["std"] + stats["mean"]
    ph = per_horizon_mae(y_true, y_pred)
    return {
        "mae": mae(y_true, y_pred),
        "per_horizon_mae": ph.tolist(),
        "weighted_horizon_score": weighted_horizon_score(ph),
        "peak_timing_error": peak_timing_error(y_true, y_pred),
        "peak_magnitude_error": peak_magnitude_error(y_true, y_pred),
        "positivity_violation": positivity_violation(y_pred),
    }


def evaluate(model, loader, device):
    """Thin MAE-only wrapper, kept name/signature-compatible with scripts/run_exp.py."""
    return evaluate_full(model, loader, device)["mae"]
