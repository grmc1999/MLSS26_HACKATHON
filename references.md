# References — Physics-Consistent Forecasting for Dynamical Systems

## Reading List for the Paper-Inspired Branch

### 1. Diffusion Models for Time Series

| Paper | Year | Key Contribution | Link |
|-------|------|------------------|------|
| Ho, J., Jain, A., & Abbeel, P. — *Denoising Diffusion Probabilistic Models* | 2020 | Foundation: forward/reverse diffusion, DDPM sampling | [arXiv:2006.11239](https://arxiv.org/abs/2006.11239) |
| Song, J., Meng, C., & Ermon, S. — *Denoising Diffusion Implicit Models* | 2020 | DDIM: deterministic sampling, 10-50x faster | [arXiv:2010.02502](https://arxiv.org/abs/2010.02502) |
| Tashiro, Y., Song, J., & Ermon, S. — *CSDI: Conditional Score-based Diffusion Models for Probabilistic Time Series Imputation* | 2021 | Conditional diffusion for time series imputation & forecasting | [arXiv:2107.03502](https://arxiv.org/abs/2107.03502) |
| Rasul, K., et al. — *Autoregressive Denoising Diffusion Models for Multivariate Probabilistic Time Series Forecasting* | 2021 | TimeGrad: diffusion for multivariate forecasting | [arXiv:2101.12072](https://arxiv.org/abs/2101.12072) |
| Bilos, M., et al. — *Scalable Transformers for Neural Probabilistic Time Series Forecasting* | 2023 | Transformer-based diffusion for long-horizon forecasting | (NeurIPS 2023) |

### 2. Neural ODEs and Continuous Dynamics

| Paper | Year | Key Contribution | Link |
|-------|------|------------------|------|
| Chen, R.T.Q., et al. — *Neural Ordinary Differential Equations* | 2018 | Continuous-depth models, adjoint sensitivity method | [arXiv:1806.07366](https://arxiv.org/abs/1806.07366) |
| Rubanova, Y., Chen, R.T.Q., & Duvenaud, D. — *Latent ODEs for Irregularly-Sampled Time Series* | 2019 | Latent ODE: continuous latent state dynamics | [arXiv:1907.03907](https://arxiv.org/abs/1907.03907) |
| Kidger, P., et al. — *Neural Controlled Differential Equations for Irregular Time Series* | 2020 | Neural CDE: controlled path for irregular data | [arXiv:2005.08926](https://arxiv.org/abs/2005.08926) |
| Dupont, E., Doucet, A., & Teh, Y.W. — *Augmented Neural ODEs* | 2019 | Augmented state space for richer dynamics | [arXiv:1904.01681](https://arxiv.org/abs/1904.01681) |

### 3. Compartmental Models for Epidemiology

| Paper | Year | Key Contribution | Link |
|-------|------|------------------|------|
| Kermack, W.O. & McKendrick, A.G. — *A Contribution to the Mathematical Theory of Epidemics* | 1927 | Foundation: SIR compartmental model | (Proc. Royal Society) |
| Reich, N.G., et al. — *Collaborative Influenza Forecasting (CDC FluSight)* | 2019 | FluSight framework, evaluation protocol | [DOI:10.1073/pnas.1813454116](https://doi.org/10.1073/pnas.1813454116) |
| Venna, S.R., et al. — *Influenza Forecasting with LSTM Networks* | 2019 | LSTM for ILI prediction from CDC data | [DOI:10.1016/j.eswa.2018.12.043](https://doi.org/10.1016/j.eswa.2018.12.043) |
| Sajid, M., et al. — *A Hybrid ARIMA-LSTM Model for Influenza Epidemics* | 2023 | Hybrid statistical + neural for influenza | (Academic Press, 2023) |
| Yang, W., et al. — *Transmission Dynamics and Forecasts of Influenza in the United States* | 2018 | Mechanistic + statistical ensemble for flu | (Epidemics, 2018) |

### 4. Parameter-Efficient Fine-Tuning (PEFT)

| Paper | Year | Key Contribution | Link |
|-------|------|------------------|------|
| Hu, E.J., et al. — *LoRA: Low-Rank Adaptation of Large Language Models* | 2021 | LoRA: low-rank decomposition, freeze base weights | [arXiv:2106.09685](https://arxiv.org/abs/2106.09685) |
| Houlsby, N., et al. — *Parameter-Efficient Transfer Learning for NLP* | 2019 | Adapters: bottleneck MLP per layer | [arXiv:1902.00751](https://arxiv.org/abs/1902.00751) |
| Lester, B., Al-Rfou, R., & Constant, N. — *The Power of Scale for Parameter-Efficient Prompt Tuning* | 2021 | Prompt tuning vs. fine-tuning | [arXiv:2104.08691](https://arxiv.org/abs/2104.08691) |
| Pfeiffer, J., et al. — *AdapterFusion: Non-Destructive Task Composition for Transfer Learning* | 2021 | Composable adapters | [arXiv:2005.00247](https://arxiv.org/abs/2005.00247) |

### 5. Physics-Informed and Adjoint Methods

| Paper | Year | Key Contribution | Link |
|-------|------|------------------|------|
| Raissi, M., Perdikaris, P., & Karniadakis, G.E. — *Physics-Informed Neural Networks* | 2019 | PINNs: PDE residuals as loss functions | [arXiv:1711.10561](https://arxiv.org/abs/1711.10561) |
| Lu, L., et al. — *DeepXDE: A Deep Learning Library for Solving Differential Equations* | 2021 | PINN library, adjoint support | [arXiv:1907.04502](https://arxiv.org/abs/1907.04502) |
| Ruthotto, L., et al. — *A Machine Learning Framework for Adjoint-Based Inference* | 2022 | Adjoint methods for neural network training | (SIAM J. Sci. Comput.) |
| Yıldız, Ç., et al. — *Adjoint Matching for Physics-Consistent Diffusion Fine-Tuning* | 2025 | Core methodology: align denoising with physical gradients | (In preparation) |
| Wang, S., et al. — *Physics-Guided Deep Learning for Dynamical Systems* | 2023 | Survey: physical constraints in neural models | [arXiv:2301.04400](https://arxiv.org/abs/2301.04400) |

### 6. Time Series Architectures

| Paper | Year | Key Contribution | Link |
|-------|------|------------------|------|
| Lim, B., et al. — *Temporal Fusion Transformers for Interpretable Multi-horizon Time Series* | 2021 | TFT: interpretable quantile forecasting with attention | [arXiv:1912.09363](https://arxiv.org/abs/1912.09363) |
| Oreshkin, B.N., et al. — *N-BEATS: Neural Basis Expansion for Interpretable Time Series* | 2020 | N-BEATS: basis decomposition, no feature engineering | [arXiv:1905.10437](https://arxiv.org/abs/1905.10437) |
| Bai, S., Kolter, J.Z., & Koltun, V. — *An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling* | 2018 | TCN: dilated causal convolutions for sequences | [arXiv:1803.01271](https://arxiv.org/abs/1803.01271) |
| Zhou, H., et al. — *Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting* | 2021 | Informer: ProbSparse attention for long sequences | [arXiv:2012.07436](https://arxiv.org/abs/2012.07436) |
| Vaswani, A., et al. — *Attention Is All You Need* | 2017 | Transformer foundation | [arXiv:1706.03762](https://arxiv.org/abs/1706.03762) |

### 7. RESPNET, ILINET, and Flu Forecasting

| Resource | Type | Description |
|----------|------|-------------|
| CDC FluView / ILINET | Data | Weekly influenza-like illness surveillance data by HHS region |
| RESPNET | Data | Respiratory pathogen surveillance (RSV, SARS-CoV-2, influenza) |
| CDC FluSight Challenges | Benchmark | Seasonal flu forecasting ensemble evaluation |
| Reich Lab — *flusight* | Code | R/Python tools for flu forecasting evaluation |
| CMU Delphi Group — *epidata* | API | Real-time epidemiological data API (ILI+, COVIDcast) |

### 8. Software and Libraries

| Tool | Purpose | Link |
|------|---------|------|
| torchdiffeq | Differentiable ODE solvers + adjoint sensitivity | [GitHub](https://github.com/rtqichen/torchdiffeq) |
| torchsde | Stochastic differential equation solvers | [GitHub](https://github.com/google-research/torchsde) |
| DeepXDE | PINN library | [GitHub](https://github.com/lululxvi/deepxde) |
| HuggingFace Diffusers | Diffusion model training & sampling | [GitHub](https://github.com/huggingface/diffusers) |
| CDC FluSight Eval | Forecasting evaluation tools | [GitHub](https://github.com/cdcepi/FluSight-forecast-eval) |

---

## How to Use This Reference

1. **For architecture decisions**: skim Section 6 (Time Series) + Section 1 (Diffusion)
2. **For physical constraints**: read Section 5 (Adjoint/PINNs) + Section 3 (Compartmental)
3. **For efficient fine-tuning**: focus on Section 4 (PEFT)
4. **For the core method**: Section 5 — adjoint matching + diffusion
5. **For evaluation**: Section 3 (FluSight protocol) + Section 7 (CDC resources)
6. **For the data**: Section 7 (RESPNET/ILINET)
