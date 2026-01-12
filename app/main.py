from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional, Dict, Any
from collections import Counter
import numpy as np, random, logging
from dataclasses import dataclass

app = FastAPI()

# CORS for local Angular dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("powerball_ai")

# -----------------------
# Models
# -----------------------
class DrawInput(BaseModel):
    # Each inner list = [white1..5, powerball] (zero-padded ok; we coerce to int)
    historical_draws: List[List[int]]

class GenerateRequest(BaseModel):
    historical_draws: List[List[int]]
    num_tickets: int = 30
    recency_decay: float = 0.97     # 0.95–0.99 typical
    alpha_smooth: float = 0.5       # Bayesian pseudo-count
    temperature: float = 0.8        # >1 = more random; <1 = peakier
    diversity_min_hamming: int = 3  # across returned tickets
    seed: Optional[int] = None

class BacktestRequest(BaseModel):
    historical_draws: List[List[int]]
    holdout: int = 10               # how many trailing draws to simulate as "future"
    num_tickets: int = 20
    recency_decay: float = 0.97
    alpha_smooth: float = 0.5
    temperature: float = 1.0
    seed: Optional[int] = None

# -----------------------
# Constants / helpers
# -----------------------
WHITE_MIN, WHITE_MAX, WHITE_COUNT = 1, 69, 5
PB_MIN, PB_MAX = 1, 26

@dataclass
class Weights:
    white: Dict[int, float]   # per-number prob for white pool (1..69)
    pb: Dict[int, float]      # per-number prob for powerball (1..26)

def _to_int_matrix(draws: List[List[int]]) -> np.ndarray:
    arr = []
    for row in draws:
        if len(row) != 6:
            raise ValueError("Each draw must contain exactly 6 numbers (5 white + 1 PB)")
        # coerce to int (zero-padded strings OK)
        arr.append([int(x) for x in row])
    return np.array(arr, dtype=int)

def _softmax(x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    x = x / max(1e-9, temperature)
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()

def _compute_weights(draws_np: np.ndarray, recency_decay=0.97, alpha=0.5, temperature=1.0) -> Weights:
    """
    Recency-weighted + long-term smoothed distributions for white and PB.
    """
    whites = draws_np[:, :5].reshape(-1)
    pbs = draws_np[:, 5]

    # Recency: newest draw has highest weight
    T = len(draws_np)
    rec_powers = np.array([recency_decay ** (T-1-i) for i in range(T)], dtype=float)

    # White recency counts
    white_rec = np.zeros(WHITE_MAX + 1, dtype=float)  # 0 unused
    for i, row in enumerate(draws_np[:, :5]):
        for n in row:
            white_rec[n] += rec_powers[i]

    # White long-term counts
    white_long = np.zeros(WHITE_MAX + 1, dtype=float)
    for n in whites:
        white_long[n] += 1.0

    # PB recency/long
    pb_rec = np.zeros(PB_MAX + 1, dtype=float)
    for i, n in enumerate(pbs):
        pb_rec[n] += rec_powers[i]
    pb_long = np.zeros(PB_MAX + 1, dtype=float)
    for n in pbs:
        pb_long[n] += 1.0

    # Normalize with smoothing to distributions
    def _norm_counts(counts: np.ndarray, lo: int, hi: int) -> np.ndarray:
        slice_ = counts[lo:hi+1]
        total = slice_.sum() + alpha * (hi - lo + 1)
        return (slice_ + alpha) / total

    p_white_rec  = _norm_counts(white_rec, WHITE_MIN, WHITE_MAX)
    p_white_long = _norm_counts(white_long, WHITE_MIN, WHITE_MAX)
    p_pb_rec     = _norm_counts(pb_rec, PB_MIN, PB_MAX)
    p_pb_long    = _norm_counts(pb_long, PB_MIN, PB_MAX)

    # Blend recency and long-term (you can tune weights)
    a, b = 1.0, 0.5
    white_raw = a * p_white_rec + b * p_white_long
    pb_raw    = a * p_pb_rec    + b * p_pb_long

    # Softmax with temperature -> final weights
    white_probs = _softmax(np.log(np.maximum(white_raw, 1e-12)), temperature=temperature)
    pb_probs    = _softmax(np.log(np.maximum(pb_raw, 1e-12)),    temperature=temperature)

    white_dict = {n: float(white_probs[n-WHITE_MIN]) for n in range(WHITE_MIN, WHITE_MAX+1)}
    pb_dict    = {n: float(pb_probs[n-PB_MIN])       for n in range(PB_MIN, PB_MAX+1)}
    return Weights(white=white_dict, pb=pb_dict)

def _sample_whites_without_replacement(weights: Dict[int, float], k=WHITE_COUNT, rng: random.Random = random) -> List[int]:
    pool = list(weights.keys())
    probs = np.array([weights[n] for n in pool], dtype=float)

    picks = []
    for _ in range(k):
        if probs.sum() <= 0:
            # uniform fallback
            remaining = [n for n in pool if n not in picks]
            choice = rng.choice(remaining)
        else:
            p = probs / probs.sum()
            idx = int(np.random.choice(len(pool), p=p))
            choice = pool[idx]
            # zero out picked weight to enforce uniqueness
            probs[idx] = 0.0
        if choice in picks:
            # in rare cases due to numerical issues; choose highest remaining prob
            remaining = [(n, w) for n, w in zip(pool, probs) if n not in picks]
            remaining.sort(key=lambda t: t[1], reverse=True)
            choice = remaining[0][0]
        picks.append(choice)

    return sorted(picks)

def _sample_powerball(weights: Dict[int, float]) -> int:
    pool = list(weights.keys())
    probs = np.array([weights[n] for n in pool], dtype=float)
    p = probs / probs.sum() if probs.sum() > 0 else None
    idx = int(np.random.choice(len(pool), p=p))
    return pool[idx]

def _hamming(a: List[int], b: List[int]) -> int:
    return len(set(a[:5]) ^ set(b[:5])) + (1 if a[5] != b[5] else 0)

def _format_play(whites: List[int], pb: int) -> Dict[str, Any]:
    whites_str = [str(n).zfill(2) for n in whites]
    pb_str = str(pb).zfill(2)
    return {
        "full_set": whites_str + [pb_str],
        "white_balls": whites_str,
        "powerball": pb_str,
    }

# -----------------------
# Endpoints
# -----------------------
@app.post("/predict")
def predict_next_draw(data: DrawInput):
    """
    Backward compatible: returns ONE play using the new weighted random generator.
    """
    try:
        draws = _to_int_matrix(data.historical_draws)
    except ValueError as e:
        return {"error": str(e)}

    # Compute blended weights
    weights = _compute_weights(draws, recency_decay=0.97, alpha=0.5, temperature=1.0)

    # Sample without replacement for whites + one PB
    whites = _sample_whites_without_replacement(weights.white, k=WHITE_COUNT)
    pb = _sample_powerball(weights.pb)

    return _format_play(whites, pb)

@app.post("/generate")
def generate(req: GenerateRequest):
    """
    Returns N weighted-random plays, diversity-guarded.
    """
    try:
        draws = _to_int_matrix(req.historical_draws)
    except ValueError as e:
        return {"error": str(e)}

    if req.seed is not None:
        random.seed(req.seed)
        np.random.seed(req.seed)

    weights = _compute_weights(
        draws,
        recency_decay=req.recency_decay,
        alpha=req.alpha_smooth,
        temperature=req.temperature,
    )

    batch = []
    attempts = 0
    max_attempts = req.num_tickets * 50
    while len(batch) < req.num_tickets and attempts < max_attempts:
        whites = _sample_whites_without_replacement(weights.white, k=WHITE_COUNT)
        pb = _sample_powerball(weights.pb)
        candidate = whites + [pb]

        if all(_hamming(candidate, t) >= req.diversity_min_hamming for t in batch):
            batch.append(candidate)
        attempts += 1

    tickets = [_format_play(t[:5], t[5]) for t in batch]
    meta = {
        "requested": req.num_tickets,
        "returned": len(tickets),
        "seed": req.seed,
        "diversity_min_hamming": req.diversity_min_hamming
    }
    return {"tickets": tickets, "meta": meta}

@app.post("/backtest")
def backtest(req: BacktestRequest):
    """
    Simple leakage-safe walk-forward:
    Train on 1..t and generate plays for t+1; count overlap hits.
    """
    try:
        draws = _to_int_matrix(req.historical_draws)
    except ValueError as e:
        return {"error": str(e)}

    if len(draws) <= req.holdout:
        return {"error": "Not enough rows to hold out."}

    if req.seed is not None:
        random.seed(req.seed)
        np.random.seed(req.seed)

    train_end = len(draws) - req.holdout
    results = []
    total_hits_whites = 0
    total_hits_pb = 0

    for t in range(train_end, len(draws)):
        train_slice = draws[:t]  # train on past only
        target = draws[t]        # the next actual draw

        weights = _compute_weights(train_slice, req.recency_decay, req.alpha_smooth, req.temperature)
        hits_w, hits_pb = 0, 0

        # generate a small batch and measure best overlap
        for _ in range(req.num_tickets):
            whites = _sample_whites_without_replacement(weights.white, k=WHITE_COUNT)
            pb = _sample_powerball(weights.pb)

            hits_w = max(hits_w, len(set(whites) & set(target[:5])))
            hits_pb = max(hits_pb, int(pb == target[5]))

        total_hits_whites += hits_w
        total_hits_pb += hits_pb
        results.append({"step": t, "white_hits": hits_w, "pb_hit": hits_pb})

    return {
        "steps": len(results),
        "summary": {"white_hits_sum": total_hits_whites, "pb_hits_sum": total_hits_pb},
        "detail": results
    }

@app.post("/train")
def train_model(data: DrawInput):
    # No persisted model yet; intentionally stateless for Angular iterations
    try:
        draws = _to_int_matrix(data.historical_draws)
    except ValueError as e:
        return {"error": str(e)}
    logger.info("Validated %d draws for training placeholder.", len(draws))
    return {"status": "ok"}
