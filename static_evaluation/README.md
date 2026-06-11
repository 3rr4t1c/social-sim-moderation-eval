# Misinformation Reshare Network Dismantling Evaluation

This tool compares dismantling strategies for misinformation reshare networks using real and synthetic data.

## Installation

```bash
pip install -r requirements.txt
```

## Project Structure

```
.
├── evaluate_dismantling.py # CLI entry point
├── requirements.txt        # Python dependencies
├── data/
│   ├── real/              # Real datasets (.csv)
│   └── synthetic/         # Synthetic simulations
│       └── *_simulation/  # Each simulation folder contains run_*.csv files
├── output/                # Generated plots (PDF)
└── src/
    ├── data_loader.py     # Data loading and preprocessing
    ├── pipeline.py        # Main evaluation pipeline
    ├── ranking/           # User ranking methods
    │   ├── superspreaders.py  # TASH-index, Social H-index, etc.
    │   ├── amplifiers.py      # Early Reposter, Repost Count, etc.
    │   ├── coordinated.py     # Cosine similarity methods
    │   ├── ml_ranking.py      # Random Forest hybrid
    │   └── utils.py           # Shared utilities
    └── evaluation/        # Dismantling and visualization
        ├── dismantling.py # Network dismantling algorithms
        └── plotting.py    # Comparison plots
```

## Usage

### Basic Usage

Run with default settings (TASH-index, Time-Aware Influential, Repost Count, Cosine Eigenvector, Mean Post Credibility):

```bash
python evaluate_dismantling.py
```

### Custom Rankers

Specify which ranking methods to evaluate:

```bash
python evaluate_dismantling.py --rankers tash_index early_reposter random_forest
```

### Custom Credibility Threshold

```bash
python evaluate_dismantling.py --cred-threshold 50.0
```

### List Available Rankers

```bash
python evaluate_dismantling.py --list-rankers
```

### Full Options

```bash
python evaluate_dismantling.py --help
```

## Data Format

### Input CSV Format

Both real and synthetic data should have these columns:

| Column | Description |
|--------|-------------|
| `action_id` | Unique action identifier |
| `timestamp` | Action timestamp |
| `author_id` | User who performed the action |
| `action_type` | "post" or "reshare" |
| `target_action_id` | ID of reshared post (for reshares) |
| `target_author_id` | Author of reshared post (for reshares) |
| `extra` | List containing credibility score, e.g., `[64.5]` |

### Directory Structure

```
data/
├── real/
│   └── dataset_name.csv
└── synthetic/
    └── simulation_name_simulation/
        ├── run_1.csv
        ├── run_2.csv
        └── run_3.csv
```

## Output

The tool generates side-by-side comparison plots:
- Left: Real data dismantling curves
- Right: Synthetic data curves with confidence intervals (mean ± std)

Output files are saved as PDF in the `output/` directory.

## Available Ranking Methods

### Superspreader Methods
- `tash_index` - Time-Aware Social H-index
- `social_h_index` - Static Social H-index
- `influential` - Total reshares received
- `time_aware_influential` - Time-aware version
- `mean_post_credibility` - Mean credibility of authored posts

### Amplifier Methods
- `repost_count` - Total reshare count
- `early_reposter` - Early resharing behavior
- `tar_index` - Time-Aware Repost index
- `node_degree` - Degree in reshare network
- `node_strength` - Weighted degree
- `self_repost` - Self-reshare count
- `mean_repost_credibility` - Mean credibility of reshared content

### Coordinated Methods
- `cosine_eigenvector` - Eigenvector centrality on similarity graph
- `cosine_max` - Maximum similarity with any user

### ML Methods
- `random_forest` - Random Forest hybrid combining multiple features

## License

MIT
