# Project Outline  
## Adjoint-Matched Fine-Tuning of Physics-Consistent Diffusion Models for Rainfall Prediction

## 1. Project Motivation

Rainfall prediction is a critical task for flood forecasting, disaster prevention, agriculture, energy planning, and urban management. Deep learning models have shown strong potential for short-term rainfall prediction, especially in nowcasting settings. However, several limitations remain:

- Deterministic models often produce blurry predictions.
- Generative models can produce sharper and more realistic forecasts but may violate physical consistency.
- Diffusion models are expensive to train from scratch.
- Fine-tuning pretrained diffusion models is more computationally efficient, but naive fine-tuning may degrade physical plausibility.
- Rainfall prediction requires strong performance on rare and extreme events, where standard pixel-wise losses are insufficient.

This project proposes an efficient fine-tuning strategy for diffusion-based rainfall prediction that uses adjoint-informed physical gradients to improve adaptation while preserving physical consistency.

---

## 2. Main Research Question

Can adjoint matching improve the fine-tuning of pretrained diffusion models for rainfall prediction by increasing predictive skill, especially Critical Success Index, while maintaining physical consistency?

The central hypothesis is:

> A diffusion rainfall model can be fine-tuned more efficiently and more physically consistently if the learned denoising updates are aligned with adjoint-informed physical correction directions.

---

## 3. Project Objectives

### 3.1 Main Objective

Develop and evaluate an adjoint-matched fine-tuning method for physics-consistent diffusion models applied to rainfall prediction.

### 3.2 Specific Objectives

- Build an MLRC-benchmark-style rainfall prediction task.
- Evaluate fine-tuning capacity under limited data and limited compute.
- Compare diffusion fine-tuning against non-diffusion baselines.
- Measure rainfall prediction quality using CSI-centered metrics.
- Measure physical consistency through differentiable physical residuals.
- Study whether adjoint matching improves heavy-rain and extreme-rain prediction.
- Design the method so it can later be extended to PDE or ODE discovery.

---

## 4. Benchmark-Style Problem Formulation

### 4.1 Task Definition

Given a sequence of past rainfall, radar, or satellite observations,

\[
x_{t-k+1:t},
\]

predict a sequence of future rainfall fields,

\[
x_{t+1:t+h}.
\]

The initial task should focus on rainfall nowcasting:

\[
\text{past frames} \rightarrow \text{future rainfall frames}.
\]

### 4.2 Recommended Initial Setting

- Input: 6 to 12 past frames.
- Output: 6 to 12 future frames.
- Temporal resolution: 5 to 10 minutes.
- Prediction horizon: 30 to 60 minutes for fast experiments.
- Spatial resolution: 48×48 or 64×64 for fast experiments.
- Dataset: sub-SEVIR, SEVIR subset, or similar radar/satellite rainfall dataset.

### 4.3 MLRC-Benchmark-Style Requirements

The benchmark should include:

- Fixed train/validation/test splits.
- Hidden test set for final evaluation.
- Standardized preprocessing.
- Fixed evaluation script.
- Baseline implementations.
- Reproducible training scripts.
- Compute-budget tracks.
- Leaderboard-style reporting.
- Main score based on CSI and fine-tuning efficiency.

---

## 5. Proposed Method

## 5.1 Conditional Diffusion Model

We assume a pretrained conditional diffusion model:

\[
\epsilon_\theta(x_\tau, c, \tau),
\]

where:

- \(x_\tau\) is the noisy rainfall prediction at diffusion step \(\tau\),
- \(c\) is the conditioning information, usually past rainfall frames,
- \(\epsilon_\theta\) is the denoising network.

The standard diffusion loss is:

\[
\mathcal{L}_{\text{diff}}
=
\mathbb{E}_{x_0,\epsilon,\tau}
\left[
\left\|
\epsilon -
\epsilon_\theta(x_\tau,c,\tau)
\right\|^2
\right].
\]

---

## 5.2 Physical Consistency Residual

A simple physical approximation for rainfall evolution is an advection-continuity residual:

\[
\frac{\partial r}{\partial t}
+
v \cdot \nabla r
-
s
\approx 0,
\]

where:

- \(r\) is rainfall intensity,
- \(v\) is an estimated velocity or motion field,
- \(s\) is a source/sink term representing rainfall growth and decay.

The physical loss can be written as:

\[
\mathcal{L}_{\text{phys}}
=
\left\|
\frac{\partial r}{\partial t}
+
v \cdot \nabla r
-
s
\right\|^2.
\]

---

## 5.3 Adjoint Matching

The adjoint direction is defined as the gradient of the physical objective with respect to the noisy or reconstructed rainfall state:

\[
g_{\text{adj}}
=
\nabla_{x_\tau}
\Phi_{\text{phys}}(\hat{x}_0(x_\tau)).
\]

The proposed adjoint-matching loss aligns the learned fine-tuning correction with the physical correction direction:

\[
\mathcal{L}_{\text{adjoint-match}}
=
\left\|
\Delta_\theta(x_\tau,c,\tau)
+
\eta_\tau g_{\text{adj}}
\right\|^2.
\]

The total fine-tuning objective is:

\[
\mathcal{L}
=
\mathcal{L}_{\text{diff}}
+
\lambda_{\text{AM}}
\mathcal{L}_{\text{adjoint-match}}
+
\lambda_{\text{phys}}
\mathcal{L}_{\text{phys}}
+
\lambda_{\text{event}}
\mathcal{L}_{\text{event}}.
\]

---

## 5.4 Parameter-Efficient Fine-Tuning

To keep experiments computationally efficient, the method should initially use parameter-efficient fine-tuning strategies:

- LoRA.
- Adapters.
- Bias-only fine-tuning.
- Last-block fine-tuning.
- Conditioning-module fine-tuning.
- FiLM modulation.
- Small residual correction modules.

The main proposed method is:

> Adjoint-matched parameter-efficient fine-tuning of pretrained rainfall diffusion models.

---

## 6. Experimental Design

# 6.1 Stage 0 — Benchmark Construction

### Goal

Build a reproducible rainfall prediction benchmark before testing the proposed method.

### Tasks

- Select dataset.
- Define input-output format.
- Define train/validation/test split.
- Define thresholds for rainfall events.
- Implement evaluation metrics.
- Implement baseline models.
- Implement training and fine-tuning scripts.
- Define compute-budget tracks.

### Deliverables

- Dataset loader.
- Baseline models.
- Evaluation script.
- Leaderboard-style metrics report.
- Reproducible experiment configuration.

---

# 6.2 Stage 1 — Fast Experiments

### Goal

Test whether fine-tuning improves rainfall prediction under limited compute.

### Dataset Setting

- Use sub-SEVIR or small SEVIR/RainBench subset.
- Resolution: 48×48 or 64×64.
- Horizon: 30 to 60 minutes.
- One GPU setting.
- Small number of fine-tuning steps.

### Baselines

- Persistence.
- Optical-flow/advection extrapolation.
- Small deterministic U-Net.
- ConvLSTM or ConvGRU.
- Frozen pretrained diffusion model.
- Naive diffusion fine-tuning.
- LoRA diffusion fine-tuning.
- Adapter diffusion fine-tuning.
- Adjoint-matched LoRA or adapter fine-tuning.

### Fine-Tuning Regimes

Use limited target-domain data:

\[
1\%,\ 5\%,\ 10\%,\ 25\%,\ 100\%.
\]

Use fixed fine-tuning budgets:

\[
100,\ 500,\ 1000,\ 5000
\]

optimization steps.

### Main Evaluation

The main plots should be:

- CSI vs. fine-tuning samples.
- CSI vs. fine-tuning steps.
- CSI vs. GPU-hours.
- Heavy-rain CSI vs. GPU-hours.
- Physical residual vs. CSI.

---

# 6.3 Stage 2 — Physics Consistency Ablation

### Goal

Determine whether adjoint matching improves performance because of physical information rather than generic regularization.

### Methods to Compare

1. No physics:

\[
\mathcal{L}
=
\mathcal{L}_{\text{diff}}.
\]

2. Naive physics penalty:

\[
\mathcal{L}
=
\mathcal{L}_{\text{diff}}
+
\lambda
\mathcal{L}_{\text{phys}}.
\]

3. Sampling-time physics guidance.

4. Training-time adjoint matching.

5. Training-time adjoint matching plus sampling-time guidance.

### Physical Metrics

- Advection residual.
- Mass or accumulated rainfall error.
- Non-negativity violation.
- Temporal smoothness.
- Spectral similarity.
- Storm-cell displacement error.

---

# 6.4 Stage 3 — Medium-Cost Benchmark Experiments

### Goal

Evaluate the method on a stronger and more credible rainfall nowcasting benchmark.

### Dataset Setting

- Larger SEVIR subset.
- Full SEVIR.
- RainBench low-resolution setting.
- Longer prediction horizon.
- Multiple random seeds.

### Models

- Persistence.
- Optical flow / PySTEPS-style baseline.
- U-Net.
- ConvLSTM / ConvGRU.
- PredRNN-style model.
- Transformer-based nowcasting model.
- Conditional latent diffusion.
- Physics-aware diffusion baseline.
- Proposed adjoint-matched diffusion fine-tuning.

### Main Questions

- Does adjoint matching improve CSI compared with naive fine-tuning?
- Does it improve heavy-rain CSI?
- Does it reduce physical residuals?
- Is it more compute-efficient?
- Does it improve domain transfer?

---

# 6.5 Stage 4 — Expensive Experiments

### Goal

Demonstrate scalability and scientific relevance.

### Dataset Setting

- Full-resolution radar or satellite-radar rainfall data.
- Longer horizons: 1 to 3 hours.
- Multi-variable conditioning.
- Optional NWP inputs.
- Ensemble probabilistic forecasts.

### Experiments

#### Spatial Transfer

Train on one region and fine-tune on another.

#### Temporal Transfer

Pretrain on earlier years and fine-tune on later years.

#### Seasonal Transfer

Pretrain on one season and fine-tune on another.

#### Extreme-Event Transfer

Fine-tune specifically on rare heavy-rain cases.

#### Sensor Transfer

Adapt between radar, satellite, and combined sensor settings.

---

## 7. Evaluation Metrics

# 7.1 Primary Metric: Critical Success Index

For rainfall threshold \(\theta\):

\[
\text{CSI}_\theta
=
\frac{TP}{TP+FP+FN}.
\]

CSI should be reported at several rainfall thresholds:

\[
\theta \in \{0.5,\ 1,\ 4,\ 8,\ 16\}\ \text{mm/h}.
\]

The benchmark score should emphasize heavy rainfall:

\[
\text{Score}
=
0.2\,\text{CSI}_{\text{light}}
+
0.3\,\text{CSI}_{\text{moderate}}
+
0.5\,\text{CSI}_{\text{heavy}}.
\]

---

# 7.2 Secondary Categorical Metrics

Probability of Detection:

\[
\text{POD}
=
\frac{TP}{TP+FN}.
\]

False Alarm Ratio:

\[
\text{FAR}
=
\frac{FP}{TP+FP}.
\]

Heidke Skill Score:

\[
\text{HSS}
=
\frac{2(TP\cdot TN-FP\cdot FN)}
{(TP+FN)(FN+TN)+(TP+FP)(FP+TN)}.
\]

---

# 7.3 Regression Metrics

- MAE.
- RMSE.
- Log-MAE.
- Weighted RMSE.
- Event-weighted MAE.
- Accumulated rainfall error.

---

# 7.4 Probabilistic Metrics

Because diffusion models are probabilistic, evaluate:

- CRPS.
- Brier score.
- Reliability diagrams.
- Ensemble spread-skill.
- Best-of-\(N\) CSI.
- Mean-of-\(N\) CSI.
- Ensemble mean RMSE.
- Ensemble diversity.

---

# 7.5 Physical Consistency Metrics

Evaluate:

\[
R_{\text{adv}}
=
\left\|
\frac{\partial r}{\partial t}
+
v\cdot \nabla r
-
s
\right\|_2.
\]

Also report:

\[
E_{\text{mass}}
=
\left|
\sum_{i,j} \hat{r}_{t+1}(i,j)
-
\sum_{i,j} r_t(i,j)
\right|.
\]

Non-negativity violation:

\[
E_{\text{neg}}
=
\sum_{i,j,t}
\max(0,-\hat{r}_{i,j,t}).
\]

Additional physical metrics:

- Temporal consistency.
- Spatial smoothness.
- Spectral energy similarity.
- Object displacement error.
- Storm growth/decay consistency.

---

## 8. Fine-Tuning Capacity Metrics

Fine-tuning capacity should measure how efficiently a method adapts to a new domain.

### 8.1 CSI Gain per Compute

\[
\text{FTC}_{\text{compute}}
=
\frac{
\Delta \text{CSI}_{\text{target}}
}{
\text{GPU-hours}
}.
\]

### 8.2 CSI Gain per Trainable Parameter

\[
\text{FTC}_{\text{params}}
=
\frac{
\Delta \text{CSI}_{\text{target}}
}{
\#\text{trainable parameters}
}.
\]

### 8.3 CSI Gain per Fine-Tuning Sample

\[
\text{FTC}_{\text{data}}
=
\frac{
\Delta \text{CSI}_{\text{target}}
}{
\#\text{target training samples}
}.
\]

### 8.4 Main Fine-Tuning Plots

- CSI vs. number of target-domain samples.
- CSI vs. fine-tuning steps.
- CSI vs. GPU-hours.
- CSI vs. trainable parameters.
- Heavy-rain CSI vs. physical residual.
- CSI improvement vs. physical-consistency improvement.

---

## 9. Main Ablation Studies

# 9.1 Fine-Tuning Strategy

Compare:

- Full fine-tuning.
- Last-block fine-tuning.
- Bias-only fine-tuning.
- LoRA.
- Adapters.
- FiLM conditioning.
- Conditioning-only tuning.
- Adjoint-matched LoRA.
- Adjoint-matched adapters.

---

# 9.2 Physics Loss Placement

Compare physics information applied:

- Only on final clean prediction.
- On intermediate denoising estimates.
- In latent space.
- On the denoising direction.
- Only at late denoising steps.
- At all denoising steps.

---

# 9.3 Adjoint Matching Variants

Compare:

- Direct gradient alignment.
- Cosine alignment.
- Magnitude matching.
- Projected adjoint matching.
- Threshold-weighted adjoint matching.
- Heavy-rain-region adjoint matching.

---

# 9.4 Physical Operator Quality

Compare adjoint matching using:

- Optical-flow velocity.
- Learned motion field.
- NWP wind field.
- Differentiable advection solver.
- No source term.
- Learned source/sink term.

---

# 9.5 Domain Shift

Evaluate transfer across:

- Region.
- Year.
- Season.
- Rainfall intensity distribution.
- Sensor type.
- Storm type.

---

# 9.6 Extreme-Rain Specialization

Fine-tune on heavy rainfall examples and evaluate:

- Heavy-rain CSI.
- Extreme-rain CSI.
- False alarm ratio.
- Probability of detection.
- Storm displacement.
- Physical residual.

---

## 10. Expected Results

The proposed method is expected to:

- Improve CSI compared with frozen pretrained diffusion models.
- Improve CSI faster than full fine-tuning.
- Improve heavy-rain CSI compared with naive diffusion fine-tuning.
- Reduce physical residuals compared with unconstrained diffusion fine-tuning.
- Require fewer trainable parameters when combined with LoRA or adapters.
- Improve transfer to new rainfall regimes.
- Provide a foundation for later PDE/ODE discovery.

---

## 11. Extension to PDE or ODE Discovery

After validating adjoint-matched fine-tuning, the physical residual can be replaced or augmented with a learnable dynamics model.

Assume rainfall evolves according to:

\[
\frac{\partial r}{\partial t}
=
\mathcal{F}_\psi(r, \nabla r, \nabla^2 r, u, q, t),
\]

where:

- \(r\) is rainfall intensity,
- \(u\) is wind or motion field,
- \(q\) represents additional atmospheric variables,
- \(\mathcal{F}_\psi\) is a learnable physical operator.

The extended goal becomes:

> Learn both a generative rainfall model and a physically meaningful evolution operator that explains the forecast trajectories.

Possible extensions:

- Sparse PDE discovery.
- Neural ODE discovery in latent space.
- Neural operator discovery.
- Differentiable advection-reaction models.
- Joint fine-tuning of diffusion model and physical dynamics.
- Interpretable rainfall evolution models.

---

## 12. Recommended First Milestone

### Milestone 1: Fast MLRC-Style Rainfall Fine-Tuning Benchmark

### Dataset

- sub-SEVIR or small SEVIR subset.

### Resolution

- 48×48.

### Horizon

- 30 to 60 minutes.

### Baselines

- Persistence.
- Optical flow.
- U-Net.
- ConvLSTM.
- Frozen diffusion.
- Naive diffusion fine-tuning.
- LoRA diffusion fine-tuning.
- Adjoint-matched LoRA fine-tuning.

### Metrics

- CSI at multiple thresholds.
- Heavy-rain CSI.
- CSI gain per GPU-hour.
- CSI gain per target-domain sample.
- Physical advection residual.

### Main Deliverable

A reproducible benchmark showing whether adjoint-matched fine-tuning improves rainfall prediction skill and physical consistency under limited data and compute.

---

## 13. Final Project Contributions

The expected contributions of the project are:

1. A benchmark-style rainfall prediction task focused on fine-tuning capacity.
2. A CSI-centered evaluation protocol for rainfall diffusion models.
3. A parameter-efficient diffusion fine-tuning framework.
4. An adjoint-matching objective for physics-consistent fine-tuning.
5. A systematic comparison with non-diffusion and diffusion baselines.
6. A compute-aware experimental design separating fast, medium, and expensive experiments.
7. A path toward PDE or ODE discovery from learned rainfall dynamics.
