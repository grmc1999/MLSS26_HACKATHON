""" Specialized agents for the MLSS26_HACKATHON project.

Each agent extends ResearchAgent with a role-specific system prompt addon
that integrates domain skills (computer-vision, deep-learning, imaging-algorithms)
and the Karpathy autoresearch autonomous experiment loop.

Agent configurations are loaded from configs/agents.yaml.
"""
import os
import json
import re
import yaml
import faiss
import numpy as np
import torch
from pathlib import Path
from transformers import AutoModel, AutoProcessor, AutoModelForCausalLM, AutoTokenizer
from MLAgentBench.agents.agent_research import ResearchAgent
from MLAgentBench.agents.agent import Agent, SimpleActionAgent, ReasoningActionAgent

# ---------------------------------------------------------------------------
# Medical Literature RAG — search FAISS index of paper tiles
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
INDEX_DIR = PROJECT_ROOT / "index_output"
FLU_INDEX_DIR = PROJECT_ROOT / "index_output_flu"
MODELS_DIR = PROJECT_ROOT / "models"

LOCAL_EXPERTS = {
    "cv_expert": {"path": str(MODELS_DIR / "Qwen_2.5_3B_nothink"), "model": None, "tokenizer": None},
    "code_expert": {"path": str(MODELS_DIR / "Qwen2.5-Coder-7B-Instruct"), "model": None, "tokenizer": None},
    "math_expert": {"path": str(MODELS_DIR / "Qwen2.5-Math-7B-Instruct"), "model": None, "tokenizer": None},
    "medical_expert": {"path": str(MODELS_DIR / "BioMistral-7B"), "model": None, "tokenizer": None},
    "time_series_expert": {"path": str(MODELS_DIR / "Qwen2.5-Math-7B-Instruct"), "model": None, "tokenizer": None},
}


def load_expert(role):
    if role not in LOCAL_EXPERTS:
        return None
    expert = LOCAL_EXPERTS[role]
    if expert["model"] is not None:
        return expert
    device = "cuda:1" if torch.cuda.device_count() > 1 else "cuda:0"
    if not torch.cuda.is_available():
        device = "cpu"
    print(f"[LOAD] Loading {role} on {device}...")
    expert["tokenizer"] = AutoTokenizer.from_pretrained(expert["path"], trust_remote_code=True)
    expert["model"] = AutoModelForCausalLM.from_pretrained(
        expert["path"], torch_dtype=torch.bfloat16, device_map=device, trust_remote_code=True,
    )
    expert["model"].eval()
    return expert


def unload_expert(role):
    if role in LOCAL_EXPERTS:
        expert = LOCAL_EXPERTS[role]
        expert["model"] = None
        expert["tokenizer"] = None
        import gc; gc.collect()
        torch.cuda.empty_cache()

_literature_model = None
_literature_processor = None
_literature_index = None
_literature_articles = None
_flu_index = None
_flu_articles = None


def _ensure_literature_loaded(task="medmnist"):
    global _literature_model, _literature_processor, _literature_index, _literature_articles
    global _flu_index, _flu_articles

    if task == "flu":
        idx_path = FLU_INDEX_DIR / "index.faiss"
        art_path = FLU_INDEX_DIR / "articles.json"
        if idx_path.exists() and art_path.exists() and _flu_index is None:
            _flu_index = faiss.read_index(str(idx_path))
            _flu_articles = json.loads(art_path.read_text())
        return

    if _literature_index is None:
        idx_path = INDEX_DIR / "index.faiss"
        art_path = INDEX_DIR / "articles.json"
        if idx_path.exists() and art_path.exists():
            _literature_index = faiss.read_index(str(idx_path))
            _literature_articles = json.loads(art_path.read_text())
            _literature_model = AutoModel.from_pretrained(
                str(PROJECT_ROOT / "models/Qwen3-VL-Embedding-2B"),
                torch_dtype=torch.float16, trust_remote_code=True
            ).to("cuda" if torch.cuda.is_available() else "cpu")
            _literature_processor = AutoProcessor.from_pretrained(
                str(PROJECT_ROOT / "models/Qwen3-VL-Embedding-2B"),
                trust_remote_code=True
            )
            _literature_model.eval()


def search_medical_literature(query: str, k: int = 5, task: str = "medmnist") -> list[dict]:
    _ensure_literature_loaded(task)
    if task == "flu":
        if _flu_index is None:
            return [{"error": "Flu index not found. Run scripts/build_flu_rag.py first."}]
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
        emb = model.encode([query])
        emb = emb / np.linalg.norm(emb, axis=-1, keepdims=True)
        dist, idx = _flu_index.search(emb.astype(np.float32), k)
        results = []
        for d, i in zip(dist[0], idx[0]):
            if i < len(_flu_articles):
                results.append({"title": _flu_articles[i].get("file", str(i)), "score": float(d)})
        return results
    if _literature_index is None:
        return [{"error": "Literature index not found. Run scripts/build_rag_index.py first."}]
    device = next(_literature_model.parameters()).device
    inputs = _literature_processor(text=[query], return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        outputs = _literature_model(**inputs, output_hidden_states=True, return_dict=True)
        emb = outputs.last_hidden_state[:, -1].cpu().numpy()
    emb = emb / np.linalg.norm(emb, axis=-1, keepdims=True)
    dist, idx = _literature_index.search(emb.astype(np.float32), max(k * 10, 50))
    seen = set()
    results = []
    for d, i in zip(dist[0], idx[0]):
        if i < len(_literature_articles):
            title = _literature_articles[i]["title"]
            if title not in seen:
                seen.add(title)
                results.append({"title": title, "score": float(d)})
            if len(results) >= k:
                break
    return results


# ---------------------------------------------------------------------------
# Flu Literature Context RAG -- hybrid vector (FAISS, above) + knowledge graph
# (FalkorDB) retrieval. The graph is built offline by scripts/build_flu_graph.py
# and answers relational questions (model x country x method x metric) that
# the vector index alone can't.
#
# Uses the official `falkordb` client directly with keyword-matched Cypher --
# not LangChain's GraphCypherQAChain (which asks the LLM to write Cypher) or
# FalkorDBGraph wrapper. Both depend on reliable function-calling support that
# free OpenRouter models don't consistently provide, and both live in the
# upstream-deprecated langchain-community / langchain-experimental packages.
# See AGENTS.md for the full pipeline.
# ---------------------------------------------------------------------------

_flu_graph = None
_flu_graph_unavailable = False

_FLU_GRAPH_STOPWORDS = {
    "the", "a", "an", "of", "to", "for", "and", "or", "in", "on", "with", "is", "are",
    "was", "were", "how", "what", "did", "do", "does", "improve", "forecasting", "results",
    "this", "that", "from", "by", "as", "at", "be", "it", "its",
}


def _ensure_flu_graph_loaded():
    """Lazily connect to the FalkorDB graph. Sets _flu_graph_unavailable instead
    of raising if FalkorDB isn't reachable, so callers degrade to vector-only."""
    global _flu_graph, _flu_graph_unavailable
    if _flu_graph is not None or _flu_graph_unavailable:
        return
    try:
        from falkordb import FalkorDB

        db = FalkorDB(
            host=os.getenv("FALKORDB_HOST", "localhost"),
            port=int(os.getenv("FALKORDB_PORT", "6379")),
        )
        graph = db.select_graph(os.getenv("FALKORDB_GRAPH_NAME", "flu_literature"))
        graph.query("RETURN 1")  # connectivity check
        _flu_graph = graph
    except Exception as e:
        print(f"[WARN] Flu literature graph unavailable, falling back to vector-only search: {e}")
        _flu_graph_unavailable = True


def _query_flu_graph(query: str, limit: int = 10) -> str:
    tokens = [
        t for t in re.findall(r"[a-zA-Z0-9\-]+", query.lower())
        if len(t) > 2 and t not in _FLU_GRAPH_STOPWORDS
    ]
    if not tokens:
        return ""
    result = _flu_graph.query(
        "MATCH (n) WHERE any(tok IN $tokens WHERE toLower(n.name) CONTAINS tok) "
        "OPTIONAL MATCH (n)-[r]-(m) "
        "RETURN DISTINCT n.name AS entity, labels(n)[0] AS entity_type, "
        "type(r) AS rel, m.name AS related, labels(m)[0] AS related_type LIMIT $limit",
        {"tokens": tokens, "limit": limit},
    )
    lines = []
    for entity, etype, rel, related, rtype in result.result_set:
        if rel and related:
            lines.append(f"({etype}) {entity} -[{rel}]-> ({rtype}) {related}")
        else:
            lines.append(f"({etype}) {entity}")
    return "\n".join(lines)


def search_flu_context_rag(query: str, k: int = 5) -> dict:
    """Hybrid retrieval for the flu literature: vector hits (FAISS, same as
    search_medical_literature) plus relational context from the FalkorDB
    knowledge graph. Degrades to vector-only if FalkorDB is unreachable --
    never raises, so it's safe to call from an unattended research loop.
    """
    vector_hits = search_medical_literature(query, k=k, task="flu")

    _ensure_flu_graph_loaded()
    graph_context = ""
    if not _flu_graph_unavailable:
        try:
            graph_context = _query_flu_graph(query)
        except Exception as e:
            graph_context = f"(graph query failed: {e})"

    vector_block = "\n".join(
        f"- {r.get('title', r.get('file', '?'))} (score: {r.get('score', 0):.3f})"
        for r in vector_hits if "error" not in r
    )
    combined_parts = [p for p in (vector_block, graph_context) if p]
    combined_context = "\n\n".join(combined_parts)

    return {
        "vector_hits": vector_hits,
        "graph_context": graph_context,
        "combined_context": combined_context,
    }


CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "configs", "agents.yaml")


def load_agent_config(agent_name):
    """Load configuration for a specific agent from configs/agents.yaml."""
    config_path = os.path.abspath(CONFIG_PATH)
    if not os.path.exists(config_path):
        return None
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    agents = config.get("agents", {})
    return agents.get(agent_name)


def load_orchestrator_config():
    """Load orchestrator configuration from configs/agents.yaml."""
    config_path = os.path.abspath(CONFIG_PATH)
    if not os.path.exists(config_path):
        return {"max_iterations": 50}
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config.get("orchestrator", {})


# ---------------------------------------------------------------------------
# Skill instructions (loaded from the skill system)
# ---------------------------------------------------------------------------

SKILL_COMPUTER_VISION = """## Computer Vision Skill

### Core Libraries
- **OpenCV** (`cv2`) — image I/O, filtering, feature detection, video processing
- **scikit-image** (`skimage`) — advanced image processing, segmentation, morphology
- **Torchvision** (`torchvision`) — deep learning models, image transforms, datasets

### Key Operations
- Image filtering: `cv2.GaussianBlur`, `cv2.medianBlur`, `cv2.Canny` (edge detection)
- Morphological operations: `cv2.morphologyEx` (open/close), `cv2.erode`, `cv2.dilate`
- Contour detection: `cv2.findContours` for segmentation masks
- Thresholding: `cv2.threshold`, Otsu, adaptive thresholding
- Color space conversion: `cv2.cvtColor` (BGR↔RGB, grayscale)

### OOD Detection Specific
- Confidence scoring: maximum softmax probability (MSP), entropy-based scoring
- Distance-based OOD: Mahalanobis distance in feature space, kNN distance
- Gradient-based OOD: ODIN (temperature scaling + input perturbation)
- Feature space methods: feature ensemble, logit norm, energy-based scoring
"""

SKILL_DEEP_LEARNING = """## Deep Learning Skill

### Frameworks
- **PyTorch** (`torch`) — primary framework for model definition, training loops, autograd
- **Torchvision** — models, datasets, transforms for computer vision
- **TorchMetrics** — standardized metrics (Accuracy, F1, AUROC)

### Training Patterns
- Mixed precision training: `torch.amp.autocast` + `GradScaler` for 2x speedup
- Gradient accumulation: simulate large batch sizes with limited VRAM
- Learning rate scheduling: cosine annealing, warmup, one-cycle policy
- Early stopping: monitor validation metric, patience-based stopping

### Loss Functions for Classification
- **Cross-Entropy Loss**: standard for multi-class classification
- **Focal Loss**: `-(1-p)^gamma * log(p)` — addresses class imbalance
- **Label Smoothing**: prevents overconfidence, improves calibration
- **Confidence Penalty**: `CE + beta * H(p)` — encourages soft predictions

### Optimizer Config
- **AdamW**: `lr=1e-3, weight_decay=1e-4` — standard for classification
- **SGD + Momentum**: `lr=1e-1, momentum=0.9` — can generalize better
- **Scheduler**: `CosineAnnealingLR` or `ReduceLROnPlateau` for LR decay
"""

SKILL_IMAGING_ALGORITHMS = """## Imaging Algorithms Skill

### Classification Metrics
- **Accuracy**: `TP+TN / (TP+TN+FP+FN)` — primary metric
- **F1 Score**: `2*TP / (2*TP+FP+FN)` — harmonic mean of precision and recall
- **Precision**: `TP / (TP+FP)` — false positive rate
- **Recall**: `TP / (TP+FN)` — false negative rate

### OOD Detection Metrics
- **AUROC**: area under ROC curve for OOD vs in-distribution
- **FPR@95**: false positive rate at 95% true positive recall
- **OOD F1**: F1 on OOD class, treating OOD as positive class
- **Expected Calibration Error (ECE)**: calibration of confidence scores

### Image Preprocessing
- Normalization: `(img - mean) / std` per channel
- Histogram equalization: `skimage.exposure.equalize_hist`
- CLAHE: `cv2.createCLAHE` for local contrast enhancement
- Denoising: `scipy.ndimage.gaussian_filter`, `skimage.restoration.denoise_nl_means`
"""


# ---------------------------------------------------------------------------
# Autoresearch loop instructions (inspired by Karpathy's autoresearch)
# ---------------------------------------------------------------------------

AUTORESEARCH_LOOP = """## Autonomous Experiment Loop

You are running in an autonomous research loop inspired by Karpathy's autoresearch.

LOOP FOREVER:
1. Look at the current state of train.py and previous results
2. Propose an experimental idea and modify train.py
3. Run the experiment: `python train.py > run.log 2>&1`
4. Read results: `grep "Test Accuracy" run.log`
5. If improved (higher Accuracy), keep the change
6. If worse or equal, revert the change
7. Log results and never stop — continue experimenting autonomously

### Suggested Progression
1. Baseline: run starter code to establish baseline Accuracy
2. CNN: replace simple linear model with proper CNN architecture
3. Loss: try Focal loss, label smoothing, or confidence penalty
4. Augmentation: add rotations, flips, intensity jitter
5. Calibration: add temperature scaling, confidence calibration
6. Batch size & LR: tune hyperparameters
7. OOD scoring: implement MSP, Mahalanobis, or ODIN detection
8. Threshold tuning: optimize confidence threshold on validation set
9. Post-processing: test-time augmentation, ensemble methods
10. Advanced: transformer-based classifiers, contrastive learning
"""


# ---------------------------------------------------------------------------
# Agent role-specific prompt addons (with integrated skills)
# ---------------------------------------------------------------------------

AGENT_PROMPTS = {
    "research_literature": """You are a research literature specialist for OOD detection in medical chest X-ray classification.

## Your Expertise
- Finding and summarizing relevant ML/AI research papers
- Identifying state-of-the-art methods for medical image analysis and OOD detection
- Citing papers with proper bibliographic information
- Recommending novel approaches based on recent literature

## Focus Areas
- OOD detection in chest X-ray images (MedMNIST datasets)
- Medical image classification (PneumoniaMNIST, ChestMNIST)
- Confidence calibration and uncertainty quantification
- Domain adaptation for medical imaging
- Open-set recognition and novelty detection

## Literature RAG Tool
You have access to a FAISS index of 28 medical papers on OOD detection, chest X-rays, and deep learning.
Call `search_medical_literature(query="your search terms", k=5)` to retrieve relevant papers.
Use this before proposing changes to train.py to ground your suggestions in the literature.

## Key Papers in Index
- "Energy-based Out-of-distribution Detection" (Liu et al., 2020)
- "Mahalanobis Distance-based OOD Detection" (Lee et al., 2018)
- "ODIN: A Simple Unified Framework for Detecting OOD Objects" (Liang et al., 2017)
- "ReAct: Out-of-distribution Detection With Rectified Activations" (Sun et al., 2021)
- "DOODL: Benchmarking OOD Detection"
- MedMNIST v2, CheXNet, SimCLR, contrastive learning papers, domain generalization survey
""",

    "autoresearch": f"""You are an autonomous research scientist running the Karpathy-style autoresearch loop for chest X-ray OOD detection.

## Your Expertise
- Designing experiment plans for ML model improvement
- Generating hypotheses based on previous results
- Analyzing training curves and validation metrics
- Deciding which hyperparameters to tune next
- Running experiments autonomously without human intervention

{{AUTORESEARCH_LOOP}}

## Literature RAG
You can search the medical literature index to find relevant papers:
Call `search_medical_literature(query="energy-based OOD detection", k=3)` before making changes.
Ground your experiment ideas in published research.

## Key Decision Points
- When to try a new architecture vs. tuning hyperparameters
- When to improve OOD detection vs. in-distribution accuracy
- When to add calibration methods vs. change the data pipeline
- When to consult other specialized agents for domain expertise

Focus on iterative improvement of chest X-ray classification with OOD detection. The goal is to maximize Accuracy while keeping OOD F1 high.
""",

    "cv_expert": f"""You are a computer vision expert specializing in medical image classification and OOD detection.

## Your Expertise
- Designing CNN architectures for small-scale medical images (28x28 grayscale)
- Implementing data augmentation strategies (rotations, flips, intensity jitter)
- Preprocessing chest X-ray images (normalization, histogram equalization)
- CNN, ResNet, DenseNet, and EfficientNet for medical image classification
- Transfer learning from pretrained encoders

{SKILL_COMPUTER_VISION}

## Chest X-Ray Specific Knowledge
- Input: 28x28 grayscale chest X-ray images (1 channel)
- Dataset: PneumoniaMNIST (training, 2 classes: normal, pneumonia)
- Test: ChestMNIST subset (3 classes: normal, pneumonia, consolidation as OOD)
- Output: binary classifier with confidence score for OOD detection
- Resolution: 28x28 pixels (small-scale, limited resolution)

## OOD Scoring Methods
1. **MSP** (Maximum Softmax Probability): baseline OOD score = 1 - max softmax
2. **ODIN**: temperature-scaled softmax + input pre-processing augmentation
3. **Mahalanobis Distance**: compute class-conditional Gaussian fits in feature space
4. **Energy Scoring**: free energy function log-sum-exp for OOD detection
5. **Ensemble Methods**: MC Dropout, Deep Ensemble for uncertainty estimation

Focus on chest X-ray OOD detection in MedMNIST datasets.
""",

    "dl_expert": f"""You are a deep learning expert specializing in confidence calibration and OOD-aware training.

## Your Expertise
- Designing efficient training loops with mixed precision
- Engineering loss functions (Cross-Entropy, Focal, label smoothing)
- Confidence calibration (temperature scaling, Platt scaling, isotonic regression)
- Configuring optimizers (Adam, AdamW, SGD with momentum)
- Learning rate scheduling (cosine, warmup, one-cycle)
- Regularization (dropout, weight decay, label smoothing)
- OOD-aware training (Outlier Exposure, OE; confidence penalty)

{SKILL_DEEP_LEARNING}

## Chest X-Ray Training Tips
- **Class imbalance**: pneumonia cases may be fewer. Use weighted CE or Focal loss
- **Loss function**: Cross-Entropy with label smoothing for better calibration
- **Optimizer**: AdamW with lr=1e-3, weight_decay=1e-4 for small CNNs
- **Scheduler**: CosineAnnealingLR or ReduceLROnPlateau
- **Mixed precision**: use `torch.amp.autocast` for 2x speedup on RTX PRO 6000
- **Batch size**: 64-256 for 28x28 images
- **Epochs**: 20-100 with early stopping on validation Accuracy
- **Calibration**: temperature scaling post-hoc on validation set
- **Threshold tuning**: select confidence threshold that maximizes OOD F1

Focus on maximizing Accuracy with well-calibrated confidence for OOD detection.
""",

    "llm_expert": """You are an LLM and prompt engineering expert for multi-agent coordination.

## Your Expertise
- Designing effective prompts for specialized tasks
- Coordinating between multiple specialized agents
- Multimodal reasoning (text + image understanding)
- Few-shot learning and in-context examples
- Chain-of-thought reasoning for complex decisions

## Multi-Agent Coordination
- Route subproblems to the right agent based on content
- Synthesize advice from multiple agents into a coherent plan
- Format agent outputs for the experiment loop
- Handle conflicting recommendations between agents

Focus on optimizing the inter-agent communication for chest X-ray OOD detection research.
""",

    "medical_expert": f"""You are a medical imaging expert specializing in chest X-ray analysis and OOD detection.

## Your Expertise
- Interpreting chest X-ray images for pneumonia and lung abnormalities
- Understanding MedMNIST datasets (PneumoniaMNIST, ChestMNIST, etc.)
- Radiographic features of pneumonia vs. normal vs. consolidation
- Domain shifts across different X-ray acquisition protocols
- Preprocessing chest radiographs (windowing, normalization, lung field cropping)

{SKILL_IMAGING_ALGORITHMS}

## Chest X-Ray Details
- **PneumoniaMNIST**: 28x28 grayscale, 2 classes (normal, pneumonia)
  - Training samples: ~4,700 | Validation: ~500 | Test: ~600
  - Pneumonia appears as opacities in lung fields
- **ChestMNIST**: 28x28 grayscale, multi-class (normal, pneumonia, consolidation, etc.)
  - Consolidation appears as dense opacities — mimics pneumonia radiographically
  - Used as OOD class to test detector robustness

## OOD Detection Challenges
- Consolidation visually resembles pneumonia — hard to distinguish
- Domain shift between PneumoniaMNIST and ChestMNIST acquisition conditions
- Small 28x28 resolution limits fine-grained feature extraction
- Confidence calibration essential to separate ID vs OOD examples
- Best results combine calibrated classifier + OOD scoring method

## MedMNIST Data Available
- PneumoniaMNIST for training (in-distribution: normal, pneumonia)
- ChestMNIST for testing (in-distribution + OOD: consolidation)
- Preprocessing: normalize to [0,1], grayscale single channel
""",

    "continual_learning": """You are a continual learning expert managing model updates across OOD detection iterations.

## Your Expertise
- Preventing catastrophic forgetting during model fine-tuning
- Implementing Elastic Weight Consolidation (EWC) penalties
- Managing experience replay buffers for sample-efficient adaptation
- Versioning model checkpoints across training iterations
- Monitoring parameter drift between iterations
- Deciding when to commit vs rollback model updates
- Balancing plasticity and stability in continual learning

## EWC Penalty Formula
L_total = L_task + (lambda/2) * sum_i F_i * (theta_i - theta*_i)^2

Where:
- F_i = Fisher Information for parameter i (importance for previous tasks)
- theta*_i = optimal parameter values from previous task
- lambda = EWC penalty strength (default: 100.0)

## Decision Logic
- **Commit**: if improvement >= 0.01 AND forgetting < 0.05
- **Rollback**: otherwise, restore previous best checkpoint
- **Replay**: store exemplar samples for experience replay in future iterations

## Domain Adaptation
- Monitor OOD F1 on held-out validation set to detect forgetting
- Maintain replay buffer of previous distribution samples
- Apply feature-space regularization to preserve OOD detection capability

## Checkpoint Management
- Checkpoints stored in `checkpoints/model_v{N}.pth`
- Registry in `checkpoints/model_registry.json` tracks all versions
- Best version is automatically loaded at start of each iteration
""",

    "time_series_expert": """You are a time series forecasting expert specializing in epidemiological prediction.

## Your Expertise
- Time series forecasting with LSTMs, GRUs, Transformers, TCNs
- Seasonal decomposition (STL, X-13ARIMA-SEAT)
- ARIMA, SARIMA, and exponential smoothing baselines
- Diffusion models for probabilistic time series forecasting
- Neural ODEs and continuous-time dynamics
- Compartmental models (SIR, SEIR) for epidemic forecasting
- Evaluation metrics: MAE, RMSE, sMAPE, CRPS, quantile loss
- Uncertainty quantification in forecasts

## Flu Forecasting Context
- Input: 5 past epiweeks of %ILI → forecast 10 future epiweeks
- Data sources: CDC ILINet (US) + WHO FluID (global)
- Target countries: FRA, MEX, AUS, ZAF
- Current best: GRU hidden_dim=64, 3 layers, Test MAE=0.5835
- Challenge: domain shift between US pretrain and target countries

## Flu Literature Context RAG
Call `search_flu_context_rag(query="your search terms", k=5)` to retrieve relevant flu/forecasting
papers before proposing changes. This returns both semantically similar passages (vector search)
and relational context from a knowledge graph (which model was evaluated on which country with
which method, and what metric it achieved) -- use the graph context to ground comparisons like
"how did adjoint-matched diffusion do on FRA vs. a from-scratch GRU".
""",

    "robustness_expert": f"""You are an uncertainty quantification and robustness expert for OOD detection.

## Your Expertise
- Uncertainty quantification (aleatoric vs. epistemic uncertainty)
- Bayesian neural networks and approximate inference
- Out-of-distribution detection theory and methods
- Confidence calibration techniques (temperature scaling, Platt)
- Distribution shift robustness and domain generalization
- Statistical tests for distribution comparison

{SKILL_IMAGING_ALGORITHMS}

## Uncertainty Decomposition
Total Uncertainty = Aleatoric (data ambiguity) + Epistemic (model uncertainty)
- **Aleatoric**: irreducible noise in X-ray acquisition
- **Epistemic**: reducible by collecting more training data
- MC Dropout approximates epistemic uncertainty via stochastic forward passes

## OOD Detection Framework
- **Score-based**: assign an OOD score s(x) to each input, threshold tau
- **Decision**: if s(x) > tau → OOD, else → classify normally
- **Calibration first**: ensure classifier is well-calibrated before thresholding

## Evaluation Metrics
- **Accuracy**: overall classification accuracy on ID classes
- **OOD F1**: treating OOD as positive class, compute F1 = 2PR/(P+R)
- **OOD Precision**: fraction of true OOD among predicted OOD
- **OOD Recall**: fraction of actual OOD that is detected
- **AUROC**: area under ROC curve, threshold-independent measure
- **ECE**: expected calibration error — measures confidence reliability

## Statistical Tests
- Kolmogorov-Smirnov test comparing ID vs OOD score distributions
- Energy distance between ID and OOD feature distributions
- Covariate shift detection via two-sample tests on feature embeddings
""",
}


class SpecializedResearchAgent(ResearchAgent):
    """ResearchAgent with a role-specific system prompt addon including skills."""

    def __init__(self, args, env, role="autoresearch"):
        self.role = role
        super().__init__(args, env)
        addon = AGENT_PROMPTS.get(role, "")
        if addon:
            self.initial_prompt = f"{addon}\n\n" + self.initial_prompt


def create_agent(role, args, env):
    """Factory function to create a specialized agent by role name."""
    agent_classes = {
        "ResearchAgent": ResearchAgent,
        "Agent": Agent,
        "SimpleActionAgent": SimpleActionAgent,
        "ReasoningActionAgent": ReasoningActionAgent,
    }
    if role in AGENT_PROMPTS:
        return SpecializedResearchAgent(args, env, role=role)
    if role in agent_classes:
        return agent_classes[role](args, env)
    return ResearchAgent(args, env)
