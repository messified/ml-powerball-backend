# ML Powerball Backend

A FastAPI-based backend service for generating Powerball number predictions using machine learning techniques. The service uses recency-weighted probability distributions and Bayesian smoothing to generate statistically-informed number combinations.

## Features

- **Single Prediction**: Generate one Powerball play based on historical draws
- **Batch Generation**: Generate multiple diverse tickets with configurable parameters
- **Backtesting**: Walk-forward validation to test prediction accuracy
- **Training Endpoint**: Placeholder for future model training capabilities

## Tech Stack

- **FastAPI**: Modern, fast web framework for building APIs
- **NumPy**: Numerical computing for probability calculations
- **Pydantic**: Data validation and settings management
- **Docker**: Containerized deployment with hot-reload support
- **Python 3.11+**: Required Python version

## Installation

### Prerequisites

- Python 3.11 or higher
- pip (Python package manager)
- Docker and Docker Compose (optional, for containerized deployment)

### Local Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd ml-powerball-backend
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the server:
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

## Quick Start

### Running with Uvicorn

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Running with Docker Compose

```bash
docker-compose up --build
```

The service will be available at `http://localhost:8000` with hot-reload enabled via volume mounts.

## API Endpoints

### POST `/predict`

Generate a single Powerball prediction based on historical draws.

**Request Body:**
```json
{
  "historical_draws": [
    [1, 5, 23, 45, 67, 12],
    [2, 8, 19, 34, 56, 15],
    ...
  ]
}
```

Each draw must contain exactly 6 numbers: 5 white balls (1-69) followed by 1 Powerball (1-26).

**Response:**
```json
{
  "full_set": ["01", "05", "23", "45", "67", "12"],
  "white_balls": ["01", "05", "23", "45", "67"],
  "powerball": "12"
}
```

### POST `/generate`

Generate multiple diverse Powerball tickets with configurable parameters.

**Request Body:**
```json
{
  "historical_draws": [[...], [...]],
  "num_tickets": 30,
  "recency_decay": 0.97,
  "alpha_smooth": 0.5,
  "temperature": 0.8,
  "diversity_min_hamming": 3,
  "seed": 42
}
```

**Parameters:**
- `historical_draws` (required): Array of historical draws
- `num_tickets` (default: 30): Number of tickets to generate
- `recency_decay` (default: 0.97): Weight decay factor for recent draws (0.95-0.99 typical)
- `alpha_smooth` (default: 0.5): Bayesian pseudo-count for smoothing
- `temperature` (default: 0.8): Sampling temperature (>1 = more random, <1 = peakier)
- `diversity_min_hamming` (default: 3): Minimum Hamming distance between tickets
- `seed` (optional): Random seed for reproducibility

**Response:**
```json
{
  "tickets": [
    {
      "full_set": ["01", "05", "23", "45", "67", "12"],
      "white_balls": ["01", "05", "23", "45", "67"],
      "powerball": "12"
    },
    ...
  ],
  "meta": {
    "requested": 30,
    "returned": 30,
    "seed": 42,
    "diversity_min_hamming": 3
  }
}
```

### POST `/backtest`

Perform walk-forward validation to test prediction accuracy.

**Request Body:**
```json
{
  "historical_draws": [[...], [...]],
  "holdout": 10,
  "num_tickets": 20,
  "recency_decay": 0.97,
  "alpha_smooth": 0.5,
  "temperature": 1.0,
  "seed": 42
}
```

**Parameters:**
- `historical_draws` (required): Array of historical draws
- `holdout` (default: 10): Number of trailing draws to simulate as "future"
- `num_tickets` (default: 20): Number of tickets to generate per step
- `recency_decay` (default: 0.97): Weight decay factor
- `alpha_smooth` (default: 0.5): Bayesian smoothing parameter
- `temperature` (default: 1.0): Sampling temperature
- `seed` (optional): Random seed for reproducibility

**Response:**
```json
{
  "steps": 10,
  "summary": {
    "white_hits_sum": 15,
    "pb_hits_sum": 2
  },
  "detail": [
    {
      "step": 100,
      "white_hits": 2,
      "pb_hit": 0
    },
    ...
  ]
}
```

### POST `/train`

Training placeholder endpoint (currently stateless).

**Request Body:**
```json
{
  "historical_draws": [[...], [...]]
}
```

**Response:**
```json
{
  "status": "ok"
}
```

## Configuration

### Key Parameters

- **`recency_decay`** (0.95-0.99): Controls how much weight recent draws have compared to older ones. Higher values (closer to 1.0) give more weight to recent draws.
- **`alpha_smooth`** (typically 0.5): Bayesian pseudo-count that prevents zero probabilities and smooths the distribution.
- **`temperature`**: Controls randomness in sampling:
  - `> 1.0`: More random, flatter distribution
  - `= 1.0`: Standard sampling
  - `< 1.0`: More focused on high-probability numbers
- **`diversity_min_hamming`**: Minimum number of differences required between generated tickets to ensure diversity.

## Development

### Local Development with Hot-Reload

The Docker Compose setup includes volume mounts for hot-reload during development:

```yaml
volumes:
  - ./app:/app/app
```

Changes to files in the `app/` directory will automatically reload the server.

### API Documentation

Once the server is running, interactive API documentation is available at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## License

[Add your license here]
