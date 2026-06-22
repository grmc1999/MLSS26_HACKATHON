# Project Outline
## Adjoint-Matched Fine-Tuning of Physics-Consistent Diffusion Models for Epidemiological Forecasting

## 1. Project Motivation

Forecasting influenza-like illness (ILI) and respiratory pathogen dynamics is critical for public health preparedness, hospital capacity planning, and intervention timing. Deep learning models have shown strong potential for ILI forecasting, especially with RESPNET and ILINET surveillance data. However, several limitations remain:

- Deterministic models often fail to capture trajectory uncertainty, especially during epidemic peaks.
- Generative (diffusion) models can produce realistic probabilistic forecasts but may violate physical consistency (e.g., negative ILI rates, non-conservation of population).
- Diffusion models are expensive to train from scratch.
- Fine-tuning pretrained diffusion models is more computationally efficient, but naive fine-tuning may degrade physical plausibility.
- Flu forecasting requires strong performance on rare events (peak timing, severity of novel strains), where standard MAE losses are insufficient.

This project proposes an efficient fine-tuning strategy for diffusion-based ILI forecasting that uses adjoint-informed physical gradients (from SIR/SEIR compartmental models) to improve adaptation while preserving physical consistency.

---

## 2. Main Research Question

Can adjoint matching improve the fine-tuning of pretrained diffusion models for flu forecasting by increasing predictive skill, especially peak timing/severity prediction, while maintaining physical consistency?

The central hypothesis is:

> A diffusion-based ILI forecasting model can be fine-tuned more efficiently and more physically consistently if the learned denoising updates are aligned with adjoint-informed physical correction directions derived from differentiable SIR/SEIR dynamics.

---

## 3. Project Objectives

### 3.1 Main Objective

Develop and evaluate an adjoint-matched fine-tuning method for physics-consistent diffusion models applied to epidemiological forecasting with RESPNET/ILINET data.

### 3.2 Specific Objectives

- Build an MLRC-benchmark-style flu forecasting task (5 past epiweeks → 10 future epiweeks).
- Evaluate fine-tuning capacity under limited data and limited compute.
- Compare diffusion fine-tuning against non-diffusion baselines (LSTM, TCN, Transformer).
- Measure forecast quality using MAE, per-horizon MAE, and peak timing/severity metrics.
- Measure physical consistency through differentiable SIR/SEIR residuals.
- Study whether adjoint matching improves peak-season prediction.
- Design the method so it can later be extended to PDE or ODE discovery for epidemiological dynamics.

---

## 4. Benchmark-Style Problem Formulation

### 4.1 Task Definition

Given a sequence of past ILI observations:

\[
x_{t-4:t},
\]

predict a sequence of future ILI rates:

\[
x_{t+1:t+10}.
\]

- **Input**: 5 past epiweeks (weekly %ILI rates)
- **Output**: 10 future epiweeks (multi-step ahead forecast)
- **Data**: CDC ILINET surveillance network (HHS regions)
- **Metric**: MAE (primary), per-horizon MAE, peak timing error (secondary)

### 4.2 MLRC-Benchmark-Style Requirements

The benchmark should include:

- Fixed train/validation/test splits (by season).
- Standardized preprocessing (normalization, Fourier features).
- Fixed evaluation script (MAE, RMSE, peak metrics).
- Baseline implementations (seasonal naive, LSTM, TCN, Transformer).
- Reproducible training scripts.
- Compute-budget tracks.
- Leaderboard-style reporting.
- Main score based on MAE and fine-tuning efficiency.

---

## 5. Proposed Method

### 5.1 Conditional Diffusion Model (Time Series)

We assume a conditional diffusion model:

\[
\epsilon_\theta(x_\tau, c, \tau),
\]

where:

- \(x_\tau\) is the noisy ILI forecast at diffusion step \(\tau\),
- \(c\) is the conditioning information (5 past epiweeks of ILI data),
- \(\epsilon_\theta\) is the denoising network (LSTM, TCN, or Transformer backbone).

The standard diffusion loss is:

\[
\mathcal{L}_{\text{diff}} 
= \mathbb{E}_{x_0,\epsilon,\tau} \left[ \left\| \epsilon - \epsilon_\theta(x_\tau, c, \tau) \right\|^2 \right].
\]

The forward process corrupts the 10-step forecast with Gaussian noise via a cosine schedule. The reverse process iteratively denoises from pure noise to a clean 10-step forecast conditioned on the past 5 epiweeks.

### 5.2 Physical Consistency Residual (SIR/SEIR)

A physical approximation for ILI dynamics is the SIR compartmental model:

\[
\frac{dI}{dt} = \beta \frac{SI}{N} - \gamma I,
\]

where:

- \(I\) is the infected (ILI) fraction,
- \(S\) is the susceptible fraction,
- \(\beta\) is the transmission rate,
- \(\gamma\) is the recovery rate.

The physical loss is:

\[
\mathcal{L}_{\text{phys}} = \left\| \frac{d\hat{I}}{dt} - \beta \frac{S\hat{I}}{N} + \gamma \hat{I} \right\|^2.
\]

For a discrete-time forecast, we use the Euler step:

\[
\Phi_{\text{SIR}}(\hat{I}_t, S_t) = \hat{I}_t + \Delta t \left( \beta \frac{S_t \hat{I}_t}{N} - \gamma \hat{I}_t \right).
\]

The residual penalizes deviation from SIR dynamics:

\[
\mathcal{L}_{\text{phys}} = \left\| \hat{I}_{t+1} - \Phi_{\text{SIR}}(\hat{I}_t, S_t) \right\|^2.
\]

### 5.3 Adjoint Matching

The adjoint direction is the gradient of the physical objective with respect to the noisy or reconstructed ILI forecast:

\[
g_{\text{adj}} = \nabla_{x_\tau} \Phi_{\text{phys}}(\hat{x}_0(x_\tau)).
\]

The adjoint-matching loss aligns the learned fine-tuning correction with the physical correction direction:

\[
\mathcal{L}_{\text{adjoint-match}} = \left\| \Delta_\theta(x_\tau, c, \tau) + \eta_\tau g_{\text{adj}} \right\|^2,
\]

where \(\Delta_\theta = \epsilon_\theta(x_\tau, c, \tau) - x_\tau\) (the denoising update vector).

The total fine-tuning objective is:

\[
\mathcal{L} = \mathcal{L}_{\text{diff}} + \lambda_{\text{AM}} \mathcal{L}_{\text{adjoint-match}} + \lambda_{\text{phys}} \mathcal{L}_{\text{phys}} + \lambda_{\text{event}} \mathcal{L}_{\text{event}}.
\]

### 5.4 Parameter-Efficient Fine-Tuning

To keep experiments computationally efficient, use:

- **LoRA**: low-rank adaptation of denoising network linear layers.
- **Adapters**: bottleneck MLP inserted per layer.
- **Bias-only / last-block fine-tuning**: minimal parameter updates.

The main proposed method is:

> Adjoint-matched parameter-efficient fine-tuning of pretrained diffusion models for ILI forecasting.

---

## 6. Experimental Design

### 6.1 Stage 0 — Benchmark Construction

**Goal**: Build a reproducible flu forecasting benchmark before testing the proposed method.

**Tasks**:
- Select dataset (CDC ILINET, RESPNET).
- Define input-output format (5→10 epiweeks).
- Define train/validation/test splits (by season, chronologically).
- Implement evaluation metrics (MAE, RMSE, peak metrics, SIR residual).
- Implement baseline models (seasonal naive, LSTM, GRU, TCN, Transformer).
- Implement training and fine-tuning scripts.
- Define compute-budget tracks.

### 6.2 Stage 1 — Fast Experiments

**Goal**: Test whether fine-tuning improves ILI forecasting under limited compute.

**Baselines**:
- Seasonal naive (repeat last season).
- LSTM / GRU / TCN.
- Transformer.
- Frozen pretrained diffusion model.
- Naive diffusion fine-tuning.
- LoRA diffusion fine-tuning.
- Adapter diffusion fine-tuning.
- Adjoint-matched LoRA / adapter fine-tuning.

**Fine-Tuning Regimes** (limited target seasons):
\[
1,\ 2,\ 3,\ 5,\ 10 \text{ seasons}.
\]

**Main Evaluation Plots**:
- MAE vs. fine-tuning samples.
- MAE vs. fine-tuning steps.
- MAE vs. GPU-hours.
- Peak timing error vs. GPU-hours.
- SIR residual vs. MAE.

### 6.3 Stage 2 — Physics Consistency Ablation

**Methods to Compare**:
1. No physics: \(\mathcal{L} = \mathcal{L}_{\text{diff}}\).
2. Naive SIR penalty: \(\mathcal{L} = \mathcal{L}_{\text{diff}} + \lambda \mathcal{L}_{\text{phys}}\).
3. Sampling-time physics guidance.
4. Training-time adjoint matching.
5. Training-time adjoint matching + sampling-time guidance.

**Physical Metrics**:
- SIR residual (dynamics consistency).
- Positivity violation (negative ILI rates).
- Seasonal peak timing error.
- Seasonal peak magnitude error.

### 6.4 Stage 3 — Medium-Cost Benchmark Experiments

**Models**:
- Seasonal naive.
- Persistence (repeat last year).
- LSTM / GRU.
- TCN.
- Transformer.
- Conditional diffusion.
- Physics-aware diffusion baseline.
- Adjoint-matched diffusion fine-tuning.

**Main Questions**:
- Does adjoint matching improve MAE compared with naive fine-tuning?
- Does it improve peak timing prediction?
- Does it reduce SIR physical residuals?
- Is it more compute-efficient?
- Does it improve season-to-season transfer?

### 6.5 Stage 4 — Expensive Experiments

**Experiments**:
- **Temporal transfer**: Train on 2010-2019, fine-tune on 2020+ (COVID-era shift).
- **Seasonal transfer**: Train on one season, fine-tune on another.
- **Regional transfer**: Train on one HHS region, fine-tune on another.
- **Extreme-season specialization**: Fine-tune specifically on pandemic seasons.
- **ODE discovery**: Learn the dynamics operator from diffusion predictions.

---

## 7. Evaluation Metrics

### 7.1 Primary Metric: MAE

\[
\text{MAE} = \frac{1}{H} \sum_{h=1}^{H} |y_{t+h} - \hat{y}_{t+h}|,
\]

where \(H = 10\) forecast horizons.

### 7.2 Per-Horizon Metrics

- MAE at week 1, 2, ..., 10 (to measure error accumulation).
- Weighted horizon score: \(S = \sum_{h=1}^{10} w_h \cdot \text{MAE}_h\) (near-term weighted higher).

### 7.3 Peak Metrics

- **Peak timing error**: \(|\text{week}_{\text{peak}} - \widehat{\text{week}}_{\text{peak}}|\) (weeks off).
- **Peak magnitude error**: \(|\text{ILI}_{\text{peak}} - \widehat{\text{ILI}}_{\text{peak}}|\) (%ILI error).
- **Seasonal baseline**: compare to naive seasonal forecast.

### 7.4 Physical Consistency Metrics

- **SIR residual**: \(\left\| \hat{I}_{t+1} - \Phi_{\text{SIR}}(\hat{I}_t, S_t) \right\|_2\).
- **Positivity violation**: \(\sum \max(0, -\hat{y})\).
- **Conservation error**: \(|N - (S+E+I+R)|\) for SEIR.

### 7.5 Fine-Tuning Capacity Metrics

- **FTC_compute**: \(\Delta \text{MAE} / \text{GPU-hour}\).
- **FTC_params**: \(\Delta \text{MAE} / \#\text{trainable parameters}\).
- **FTC_data**: \(\Delta \text{MAE} / \#\text{target seasons}\).

---

## 8. Main Ablation Studies

### 8.1 Fine-Tuning Strategy
Full fine-tuning vs. last-block vs. bias-only vs. LoRA vs. adapters vs. adjoint-matched LoRA.

### 8.2 Physics Loss Placement
On final prediction vs. intermediate denoising estimates vs. latent space vs. denoising direction.

### 8.3 Adjoint Matching Variants
Direct gradient alignment vs. cosine alignment vs. magnitude matching vs. projected adjoint.

### 8.4 Physical Operator Quality
SIR vs. SEIR vs. SEIRD vs. learned SIR-net vs. no source term.

### 8.5 Domain Shift
Temporal (season-to-season, year-to-year), regional (HHS region transfer), pandemic shift.

---

## 9. Expected Results

The proposed method is expected to:

- Improve MAE compared with frozen pretrained diffusion models.
- Improve MAE faster than full fine-tuning.
- Improve peak timing and severity prediction compared with naive diffusion fine-tuning.
- Reduce SIR physical residuals compared with unconstrained diffusion fine-tuning.
- Require fewer trainable parameters when combined with LoRA.
- Improve transfer to new seasons and regions.
- Provide a foundation for later PDE/ODE discovery of epidemiological dynamics.

---

## 10. Extension to ODE Discovery

After validating adjoint-matched fine-tuning, the SIR physical residual can be replaced with a learnable dynamics model:

\[
\frac{dI}{dt} = \mathcal{F}_\psi(I, S, \text{covariates}, t),
\]

where \(\mathcal{F}_\psi\) is a learnable physical operator (neural ODE, symbolic regression).

The extended goal is: > Learn both a generative ILI forecast model and a physically meaningful evolution operator that explains the epidemic trajectory.

---

## 11. Recommended First Milestone

### Milestone 1: Fast MLRC-Style Flu Forecasting Benchmark

- **Dataset**: CDC ILINET (national or HHS region), 5→10 epiweeks.
- **Metrics**: MAE (primary), per-horizon MAE, peak timing error.
- **Baselines**: seasonal naive, LSTM, TCN, Transformer, diffusion.
- **Main Deliverable**: A reproducible benchmark showing whether adjoint-matched fine-tuning improves flu forecasting skill and physical consistency under limited data and compute.
