"""Flu/ILI forecasting models: baselines + the proposed adjoint-matched diffusion method.

This is the file meant to be iterated on by the autoresearch loop; env/data.py and
env/eval.py are the frozen data/metrics layers (program.md's contract).

Loader batches are 5-tuples: (x, y, S, N, naive) -- see env/data.py.
"""
import copy
import functools
import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model

try:  # cwd=env/ (scripts/run_exp.py inserts env/ onto sys.path directly)
    from data import load_pretrain_data, load_finetune_data
    from eval import evaluate, evaluate_full, sir_residual
except ImportError:  # imported as a package, e.g. `from env.train import ...`
    from env.data import load_pretrain_data, load_finetune_data
    from env.eval import evaluate, evaluate_full, sir_residual

prepare_data = load_pretrain_data  # scripts/run_exp.py imports `prepare_data` by this name

INPUT_STEPS = 5
FORECAST_STEPS = 10

# --------------------------------------------------------------------------
# Baselines (cover all of scripts/run_exp.py's existing --model choices)
# --------------------------------------------------------------------------


class LSTMSeq2Seq(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=128, num_layers=2, forecast_steps=FORECAST_STEPS):
        super().__init__()
        self.forecast_steps = forecast_steps
        self.encoder = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.decoder = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.out_proj = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        _, (h, c) = self.encoder(x)
        dec_input = x[:, -1:, :]
        outputs = []
        for _ in range(self.forecast_steps):
            out, (h, c) = self.decoder(dec_input, (h, c))
            pred = self.out_proj(out)
            outputs.append(pred)
            dec_input = pred
        return torch.cat(outputs, dim=1)


class GRUSeq2Seq(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=128, num_layers=2, forecast_steps=FORECAST_STEPS):
        super().__init__()
        self.forecast_steps = forecast_steps
        self.encoder = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.decoder = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.out_proj = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        _, h = self.encoder(x)
        dec_input = x[:, -1:, :]
        outputs = []
        for _ in range(self.forecast_steps):
            out, h = self.decoder(dec_input, h)
            pred = self.out_proj(out)
            outputs.append(pred)
            dec_input = pred
        return torch.cat(outputs, dim=1)


class TCNForecaster(nn.Module):
    """Dilated causal Conv1d stack over the input window, projected to the forecast horizon."""

    def __init__(self, input_dim=1, hidden_dim=64, num_layers=3, forecast_steps=FORECAST_STEPS, input_steps=INPUT_STEPS):
        super().__init__()
        self.forecast_steps, self.input_dim = forecast_steps, input_dim
        layers, in_ch = [], input_dim
        for i in range(num_layers):
            dilation = 2 ** i
            layers += [nn.Conv1d(in_ch, hidden_dim, kernel_size=3, padding=dilation, dilation=dilation), nn.ReLU()]
            in_ch = hidden_dim
        self.conv = nn.Sequential(*layers)
        self.head = nn.Linear(hidden_dim * input_steps, forecast_steps * input_dim)

    def forward(self, x):
        h = self.conv(x.transpose(1, 2))[:, :, : x.shape[1]].flatten(1)
        return self.head(h).view(x.shape[0], self.forecast_steps, self.input_dim)


class TinyTransformer(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=64, num_layers=2, forecast_steps=FORECAST_STEPS, input_steps=INPUT_STEPS):
        super().__init__()
        self.forecast_steps, self.input_dim = forecast_steps, input_dim
        self.in_proj = nn.Linear(input_dim, hidden_dim)
        layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=4, dim_feedforward=hidden_dim * 2, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.head = nn.Linear(hidden_dim * input_steps, forecast_steps * input_dim)

    def forward(self, x):
        h = self.encoder(self.in_proj(x))
        return self.head(h.flatten(1)).view(x.shape[0], self.forecast_steps, self.input_dim)


# --------------------------------------------------------------------------
# Proposed method: conditional diffusion forecaster + adjoint-matched fine-tuning
# --------------------------------------------------------------------------


def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps) / timesteps
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clamp(betas, 1e-4, 0.999)


def sinusoidal_embedding(timesteps, dim):
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=timesteps.device).float() / half)
    args = timesteps.float().unsqueeze(-1) * freqs.unsqueeze(0)
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class ConditionalDenoiser(nn.Module):
    """<1M-param 1D-conv denoiser, conditioned on the 5-step input + diffusion timestep.
    cond_proj/time_proj/out_proj are deliberately plain named nn.Linear layers so
    they're clean LoRA injection targets for `apply_lora`."""

    def __init__(self, forecast_steps=FORECAST_STEPS, input_steps=INPUT_STEPS, hidden_dim=64, time_dim=32, dropout=0.15):
        super().__init__()
        self.forecast_steps = forecast_steps
        self.time_dim = time_dim
        self.cond_proj = nn.Linear(input_steps, hidden_dim)
        self.time_proj = nn.Linear(time_dim, hidden_dim)
        self.conv_in = nn.Conv1d(1, hidden_dim, kernel_size=3, padding=1)
        self.conv_mid = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.out_proj = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_tau, c, tau):
        h = self.conv_in(x_tau.transpose(1, 2))
        cond = self.cond_proj(c.squeeze(-1)).unsqueeze(-1)
        temb = self.time_proj(sinusoidal_embedding(tau, self.time_dim)).unsqueeze(-1)
        h = F.relu(h + cond + temb)
        h = self.dropout(h)
        h = F.relu(self.conv_mid(h))
        return self.out_proj(h.transpose(1, 2))


class DiffusionForecaster(nn.Module):
    """Section 5.1's epsilon_theta(x_tau, c, tau): cosine-schedule DDPM over the
    10-step forecast, conditioned on the 5-step input."""

    def __init__(self, input_dim=1, hidden_dim=64, forecast_steps=FORECAST_STEPS, input_steps=INPUT_STEPS, num_diffusion_steps=30):
        super().__init__()
        assert input_dim == 1, "denoiser assumes a univariate %ILI signal"
        self.num_diffusion_steps = num_diffusion_steps
        self.denoiser = ConditionalDenoiser(forecast_steps, input_steps, hidden_dim)
        betas = cosine_beta_schedule(num_diffusion_steps)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", torch.cumprod(1.0 - betas, dim=0))

    def q_sample(self, x0, tau, noise):
        a_bar = self.alphas_cumprod[tau].view(-1, 1, 1)
        return torch.sqrt(a_bar) * x0 + torch.sqrt(1 - a_bar) * noise

    def predict_x0(self, x_tau, c, tau):
        eps_hat = self.denoiser(x_tau, c, tau)
        a_bar = self.alphas_cumprod[tau].view(-1, 1, 1)
        return (x_tau - torch.sqrt(1 - a_bar) * eps_hat) / torch.sqrt(a_bar)

    @torch.no_grad()
    def sample(self, c):
        B, device = c.shape[0], c.device
        x = torch.randn(B, self.denoiser.forecast_steps, 1, device=device)
        for t in reversed(range(self.num_diffusion_steps)):
            tau = torch.full((B,), t, device=device, dtype=torch.long)
            eps_hat = self.denoiser(x, c, tau)
            beta_t, a_bar_t = self.betas[t], self.alphas_cumprod[t]
            a_t = 1 - beta_t
            mean = (x - beta_t / torch.sqrt(1 - a_bar_t) * eps_hat) / torch.sqrt(a_t)
            x = mean if t == 0 else mean + torch.sqrt(beta_t) * torch.randn_like(x)
        return x


def diffusion_loss(model, x0, c, device):
    """Section 5.1: L_diff = E[||eps - eps_theta(x_tau,c,tau)||^2]."""
    B = x0.shape[0]
    tau = torch.randint(0, model.num_diffusion_steps, (B,), device=device)
    noise = torch.randn_like(x0)
    x_tau = model.q_sample(x0, tau, noise)
    eps_hat = model.denoiser(x_tau, c, tau)
    return F.mse_loss(eps_hat, noise)


def physical_loss(x0_hat, S_forecast, N, beta, gamma):
    """Differentiable SIR residual (shares the exact formula used for the
    reported `sir_residual` metric in env/eval.py)."""
    return sir_residual(x0_hat.squeeze(-1), S_forecast, N, beta, gamma)


def adjoint_matching_loss(model, x_tau, c, tau, S_forecast, N, beta, gamma):
    """Section 5.3: g_adj = grad_{x_tau} Phi_phys(x0_hat(x_tau)); aligns the
    denoising update Delta_theta = eps_theta - x_tau with -eta_tau * g_adj."""
    x_tau = x_tau.clone().requires_grad_(True)
    x0_hat = model.predict_x0(x_tau, c, tau)
    phi_phys = physical_loss(x0_hat, S_forecast, N, beta, gamma)
    (g_adj,) = torch.autograd.grad(phi_phys, x_tau, create_graph=True)
    eps_hat = model.denoiser(x_tau, c, tau)
    delta_theta = eps_hat - x_tau
    eta_tau = torch.sqrt(1 - model.alphas_cumprod[tau]).view(-1, 1, 1)
    return F.mse_loss(delta_theta + eta_tau * g_adj, torch.zeros_like(delta_theta))


def peak_event_loss(x0_hat, y):
    """L_event: weighted MAE at the true peak timestep (Project_definition.md
    names but doesn't formula-define this term; this is the simplest faithful
    reading -- directly targets rare/peak-severity prediction)."""
    peak_idx = torch.argmax(y.squeeze(-1), dim=1, keepdim=True)
    true_peak = torch.gather(y.squeeze(-1), 1, peak_idx).squeeze(1)
    pred_peak = torch.gather(x0_hat.squeeze(-1), 1, peak_idx).squeeze(1)
    return F.l1_loss(pred_peak, true_peak)


def total_finetune_loss(model, x0, c, S_forecast, N, device, beta, gamma, lam_am=1.0, lam_phys=0.1, lam_event=0.1):
    """Section 5.3: L = L_diff + lam_AM*L_adjoint_match + lam_phys*L_phys + lam_event*L_event."""
    B = x0.shape[0]
    tau = torch.randint(0, model.num_diffusion_steps, (B,), device=device)
    noise = torch.randn_like(x0)
    x_tau = model.q_sample(x0, tau, noise)

    l_diff = F.mse_loss(model.denoiser(x_tau, c, tau), noise)
    l_adj = adjoint_matching_loss(model, x_tau, c, tau, S_forecast, N, beta, gamma)
    x0_hat = model.predict_x0(x_tau, c, tau)
    l_phys = physical_loss(x0_hat, S_forecast, N, beta, gamma)
    l_event = peak_event_loss(x0_hat, x0)

    return l_diff + lam_am * l_adj + lam_phys * l_phys + lam_event * l_event


def apply_lora(model, r=4, alpha=8, target_modules=("cond_proj", "time_proj", "out_proj")):
    config = LoraConfig(r=r, lora_alpha=alpha, target_modules=list(target_modules))
    model.denoiser = get_peft_model(model.denoiser, config)
    return model


def diffusion_loss_step(model, x, y, S, N):
    return diffusion_loss(model, y, x, x.device)


def adjoint_matched_loss_step(model, x, y, S, N, beta, gamma):
    S_forecast = S[:, -y.shape[1] :]
    return total_finetune_loss(model, y, x, S_forecast, N, x.device, beta, gamma)


FINETUNE_VARIANTS = ["frozen", "naive_full_finetune", "lora_no_physics", "lora_adjoint_matched", "from_scratch"]


def prepare_finetune_variant(pretrained_model, variant, hidden_dim=64, forecast_steps=FORECAST_STEPS, input_steps=INPUT_STEPS):
    if variant == "from_scratch":
        return DiffusionForecaster(hidden_dim=hidden_dim, forecast_steps=forecast_steps, input_steps=input_steps)
    if variant not in FINETUNE_VARIANTS:
        raise ValueError(f"Unknown fine-tune variant: {variant}")
    model = copy.deepcopy(pretrained_model)
    if variant == "frozen":
        model.requires_grad_(False)
    elif variant == "naive_full_finetune":
        model.requires_grad_(True)
    else:  # lora_no_physics, lora_adjoint_matched
        model.requires_grad_(False)
        model = apply_lora(model)
    return model


def get_finetune_loss_fn(variant, beta, gamma):
    if variant == "frozen":
        return None
    if variant == "lora_adjoint_matched":
        return functools.partial(adjoint_matched_loss_step, beta=beta, gamma=gamma)
    return diffusion_loss_step  # naive_full_finetune, lora_no_physics, from_scratch


# --------------------------------------------------------------------------
# Shared training utilities
# --------------------------------------------------------------------------


def create_model(model_type, input_dim=1, hidden_dim=128, num_layers=2, forecast_steps=FORECAST_STEPS):
    if model_type == "lstm":
        return LSTMSeq2Seq(input_dim, hidden_dim, num_layers, forecast_steps)
    if model_type == "gru":
        return GRUSeq2Seq(input_dim, hidden_dim, num_layers, forecast_steps)
    if model_type == "tcn":
        return TCNForecaster(input_dim, hidden_dim, num_layers, forecast_steps)
    if model_type == "transformer":
        return TinyTransformer(input_dim, hidden_dim, num_layers, forecast_steps)
    if model_type == "diffusion":
        return DiffusionForecaster(input_dim, hidden_dim, forecast_steps=forecast_steps)
    raise ValueError(f"Unknown model_type: {model_type}")


def train_epoch(model, loader, optimizer, device, loss_fn=None):
    model.train()
    total_loss, n = 0.0, 0
    for x, y, S, N, _naive in loader:
        x, y, S, N = x.to(device), y.to(device), S.to(device), N.to(device)
        optimizer.zero_grad()
        if loss_fn is not None:
            loss = loss_fn(model, x, y, S, N)
        else:
            loss = F.l1_loss(model(x), y)
        loss.backward()
        optimizer.step()
        total_loss += float(loss) * x.shape[0]
        n += x.shape[0]
    return total_loss / max(n, 1)


def build_run_record(model_name, hidden_dim, num_layers, epochs, lr, batch, model, elapsed_s, best_val_mae, best_epoch, test_mae):
    return {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": model_name,
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
        "epochs": epochs,
        "lr": lr,
        "batch": batch,
        "params": sum(p.numel() for p in model.parameters()),
        "elapsed_s": round(elapsed_s, 1),
        "best_val_mae": round(best_val_mae, 6),
        "best_epoch": best_epoch,
        "test_mae": round(test_mae, 6),
    }


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, test_loader = load_pretrain_data(batch_size=64)
    model = create_model("lstm", hidden_dim=128, num_layers=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    best_mae = float("inf")
    for _epoch in range(20):
        train_epoch(model, train_loader, optimizer, device)
        best_mae = min(best_mae, evaluate(model, val_loader, device))
    test_mae = evaluate(model, test_loader, device)
    print(f"Test MAE: {test_mae:.4f}")
