# social-sim-moderation-eval

A comparative evaluation framework to benchmark social media moderation
strategies, analyzing the gap between legacy static data and dynamic synthetic
simulations.

The repository contains two complementary, independent pipelines:

| Sub-project | Question it answers |
|-------------|---------------------|
| [`static_evaluation/`](static_evaluation/) | Given a *fixed* reshare network, how effectively does each user-ranking strategy dismantle the misinformation-spreading structure? (network-dismantling curves on real vs. synthetic data) |
| [`dynamic_evaluation/`](dynamic_evaluation/) | When a ranking strategy is applied *inside* a running simulation (live banning) vs. retroactively (static removal), how does low-quality content actually evolve before and after moderation? |

Each sub-project is self-contained, with its own `requirements.txt`, entry point
and `README`/usage notes.

## Repository layout

```
.
├── static_evaluation/        # Network-dismantling evaluation (see its README)
│   ├── evaluate_dismantling.py
│   ├── src/
│   ├── data/                 # input datasets — NOT versioned (see below)
│   └── output/               # generated PDFs — NOT versioned
├── dynamic_evaluation/       # Static-vs-dynamic moderation comparison
│   ├── evaluate_dynamic.py
│   ├── src/
│   ├── input/                # input simulations — NOT versioned
│   └── output/               # generated figures / .tex / .gml — NOT versioned
├── LICENSE                   # MIT
└── README.md
```

## Quick start

```bash
# Static dismantling evaluation
cd static_evaluation
pip install -r requirements.txt
python evaluate_dismantling.py --help

# Dynamic moderation comparison
cd dynamic_evaluation
pip install -r requirements.txt
python evaluate_dynamic.py --help
```

See [`static_evaluation/README.md`](static_evaluation/README.md) for the full
list of ranking methods, the expected CSV schema, and CLI options.

## Data and outputs

Input datasets (several GB of real and synthetic simulation traces) and the
generated outputs (figures, LaTeX tables, GML graphs) are **intentionally not
committed** — they exceed sensible git sizes and are reproducible from the code.
They are excluded via `.gitignore` and kept locally / shared out of band.

Expected local layout:

```
static_evaluation/data/{real,synthetic}/...
dynamic_evaluation/input/synt_data/<network>_day<t_mod>_top<k>_<mod_type>/<method>/<run>/activities.csv
```

## License

Released under the MIT License — see [`LICENSE`](LICENSE).
