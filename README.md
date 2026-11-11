# How much is an aggregate yield label worth?

Code for the paper *How much is an aggregate yield label worth? Pixel yield maps and their uncertainty from field, farm and regional statistics, validated against dense harvester data* (Cao Vu Bui, Hoang-Anh-Vu Nguyen and Van-Nam Nguyen; manuscript in preparation).

Harvester yield maps from [YieldSAT](https://yieldsat.github.io) serve as dense truth to measure two things:

- how much of the within-field yield pattern a pixel model learns from field, farm-year and regional yield aggregates;
- whether pixel-level uncertainty statements remain valid when they are aggregated to fields and farms.

## Experiments

| Step | Script | What it does |
|---|---|---|
| E0 | `src/e0_footprints.py` | Reconstructs physical fields from label masks and groups farms that share land into blocks, which are the leak-free split units. |
| E0 | `src/e0_features.py` | Builds a per-pixel feature table: Sentinel-2 series on relative season time, soil, terrain and season weather. Coordinates are never stored. |
| E0 | `src/e2_bags.py` | Builds pseudo-region bags: grid cells of 10, 20 and 50 km, crossed with crop and year. Only bag IDs are written. |
| E0 | `src/e6_fields.py` | Extracts the farmer-reported yields from the record metadata. |
| E1 | `src/e1_oracle.py` | Trains the dense-label oracle (LightGBM) under three split protocols: field ID, physical field and block. |
| E2 | `src/e2_aggregate.py` | Learns from aggregate labels at the field, farm-year, pseudo-region and random-bag levels, in prediction mode (P) and disaggregation mode (D). Methods: instance-level loss (B1), bag-level loss (B2), ecological ridge (B8), the same restricted to within-field covariates (B2pv, B8pv), NDVI pseudo-labels (B11) and NDVI allocation (B6). |
| E3 | `src/e3_dense_k.py` | Computes the label-value curve, i.e. skill with dense labels on *k* fields, and the transfer of dense models between countries. |
| E4/E5 | `src/e45_uq.py` | Calibrates pixel intervals with conformal prediction and aggregates the uncertainty to field and farm-year means under five constructions. Also covers calibration from aggregates only (transfer of κ) and calibration with *k* dense fields. |
| E6 | `src/e6_reported.py` | Uses farmer-reported yields as the aggregate in mode D. |
| Checks | `src/e2_blockstats.py` | Writes per-block sufficient statistics of within-field R², the input of the block bootstrap. |
| Checks | `src/e2_trainfit_check.py` | Re-trains B1, B2 and B2pv on one fold and checks that they fit the training aggregates. |
| Checks | `src/fig_examples_extract.py` | Selects example fields for the qualitative figure by a fixed rule (pixel grid, no coordinates). |

Scripts in `analysis/`:

- `export_results.py` collects the raw outputs into per-seed summaries in `results/`.
- `bootstrap_e2.py` computes block-bootstrap intervals for within-field R² and for differences between methods (`results/e2ci/`).
- `make_tables.py` writes the LaTeX tables of the paper to `outputs/tables/`.
- `make_figures.py` writes the figures to `outputs/figures/`.

## Data

YieldSAT is not redistributed here. Request it from its publishers at <https://yieldsat.github.io> and respect its license, which allows non-commercial academic use only. The scripts expect the raw folders as distributed: `<country>/<record>/` with `metadata-*.json`, `yield_masks/`, `s2_images/`, `soil/`, `dem/` and `weather/`.

Field coordinates are sensitive under the dataset's datasheet. The code never uses them as features and never writes them to disk: pseudo-region centroids stay in memory, and pixel positions are row and column indices within each record's own raster.

## Results without the data

`results/` holds the per-seed summaries behind every number, table and figure in the paper: 3 seeds and block split, with metric values only. To rebuild the tables and figures without the data, run:

```bash
python analysis/make_tables.py
python analysis/make_figures.py
```

The example-field figure is the exception, because its input holds yield maps from the dataset; it is produced only by a full run.

## Full run

```bash
conda env create -f environment.yml
conda activate agg-uq
RAW=/path/to/yieldsat/raw bash run_all.sh
```

Optional variables:

| Variable | Default | Meaning |
|---|---|---|
| `RUNS` | `./runs` | Output directory, about 15 GB in total. |
| `NTHREADS` | 16 | LightGBM threads. |
| `NWORKERS` | 16 | Processes for feature extraction. |
| `SEEDS` | `0,1,2` | Seeds to run. |

Each step skips outputs that already exist, so an interrupted run can simply be restarted. E3 also saves a checkpoint after each seed.

On a 24-core desktop CPU with 32 GB RAM, E2 takes most of a day; the other steps take a few hours each. The scripts are run with `python -I` and do not import scikit-learn or SciPy. Their wheels ship a second OpenMP runtime, and two runtimes in one process with LightGBM caused heap corruption in our runs.

## Layout

```text
src/        experiment scripts (E0-E6)
analysis/   summaries, tables, figures
results/    per-seed summaries used in the paper
run_all.sh  full pipeline
```

## Citation

If you use this code, please cite the paper (reference to be added on publication) and the dataset:

```bibtex
@inproceedings{miranda2026yieldsat,
  title     = {{YieldSAT}: A Multimodal Benchmark Dataset for High-Resolution Crop Yield Prediction},
  author    = {Miranda, Miro and Pathak, Deepak and Helber, Patrick and Bischke, Benjamin and Najjar, Hiba and Mena, Francisco and Sanchez, Cristhian and Pai, Akshay and Arenas, Diego and Valdenegro-Toro, Matias and Charfuelan, Marcela and Nuske, Marlon and Dengel, Andreas},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages     = {22920--22930},
  year      = {2026}
}
```

## License

The code is released under the MIT License (see `LICENSE`). The license does not extend to the YieldSAT data.
