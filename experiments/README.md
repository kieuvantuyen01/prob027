# Alien Tiles benchmark generation

The benchmark generator uses only the Python standard library. It creates two
families of instances:

- `witness`: a sampled click matrix is converted to a target, so feasibility is
  guaranteed. `known_upper_bound` is not an optimum label.
- `uniform`: target cells are sampled uniformly and feasibility remains
  `unclassified` until an exact solver runs.

Instances are split into `pilot` and `evaluation`. Use the pilot split while
selecting encodings or solver parameters; reserve evaluation for the final
reported experiment.

Generate the included 36-instance demo:

```bash
python3 experiments/generate_instances.py \
  --output-dir data/generated_demo \
  --sizes 3 4 --colours 2 3 \
  --densities 0.15 0.40 \
  --pilot-count 1 --evaluation-count 2 \
  --master-seed 270027
```

Generate the recommended 720-instance thesis benchmark:

```bash
python3 experiments/generate_instances.py \
  --output-dir data/benchmark_v1 \
  --sizes 4 5 6 8 10 12 --colours 2 3 4 \
  --densities 0.10 0.30 0.60 \
  --pilot-count 2 --evaluation-count 8 \
  --master-seed 270027
```

The output directory must not already exist. Validate a generated dataset with:

```bash
python3 experiments/validate_dataset.py data/generated_demo
```

Run the minimisation solver recursively over a split with:

```bash
python3 models/sat_variant2.py --input-dir data/generated_demo/pilot
```

The generator writes `manifest.csv`, `dataset_summary.json`, a dataset README,
and one JSON file per instance. Per-instance seeds and sampling use SHA-256, so
the output is independent of Python's `random.Random` implementation.

Run the generator tests with:

```bash
python3 -m unittest discover -s tests -v
```
