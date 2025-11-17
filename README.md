# Real Estate Analytics

A compact repository for exploratory data analysis, feature engineering, and simple ML models for real-estate price analytics and recommendations. The project contains Jupyter notebooks used for EDA and model building, a small Streamlit app for interactive exploration and prediction, and several datasets used during analysis.

## Highlights
- Fast EDA and profiling using pandas / ydata-profiling.
- Streamlit UI for quick demos and an interactive price predictor.
- Notebooks covering data cleaning, outlier treatment, feature selection and model selection.

## Repository structure

Top-level files and folders you'll find in the repo:

```
app.py                     # Streamlit entry point (page config + app launcher)
pyproject.toml             # Project metadata + dependencies (Python >= 3.11)
README.md                  # This file
datasets/                  # CSV datasets used across notebooks and app
pages/                     # Streamlit page modules (analysis, predictor, recommender)
*.ipynb                    # Notebooks for EDA, preprocessing and modeling
```

Example files in `datasets/`:
- `2025-11-02T23-38_export.csv`
- `Apartments.csv`
- `Data Visualization 1.csv`
- `gurgaon_properties_v1.csv`
- `gurgoan_properties.csv`
- `missing_value_imputed.csv`
- `outlier_treated.csv`
- `post_feature_selection_v2.csv`
- `post_feature_selection.csv`

Notebooks (representative):
- `EDA.ipynb`, `EDA_2.ipynb`, `EDA_profiling.ipynb` — exploratory data analysis and profiling
- `missing_value.ipynb` — missing value imputation experiments
- `outlier_treatment.ipynb` — outlier detection & handling
- `feature_selection.ipynb` — feature engineering & selection
- `model_selection.ipynb`, `baseline_model.ipynb` — model experiments and evaluation
- `recommender.ipynb` — recommender system experiments

Streamlit pages stored in `pages/`:
- `analysis_app.py` — exploratory dashboards and visualizations
- `price_predictor.py` — interactive price prediction UI
- `recommender_system.py` — property recommender demo

## Requirements
- Python 3.11 or newer (pyproject specifies `requires-python = ">=3.11"`).
- Key dependencies (listed in `pyproject.toml`): pandas, numpy, scikit-learn, streamlit, xgboost, shap, seaborn, plotly, ydata-profiling (pandas-profiling), category-encoders, and others. See `pyproject.toml` for the full list.

## Quickstart (macOS / zsh)

1. Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

2. Install the package and dependencies.

Option A — install editable from the project (recommended for development):

```bash
pip install -e .
```

Option B — if you'd prefer a plain requirements file, create one first from `pyproject.toml` or install the packages listed in `pyproject.toml` individually.

3. Run the Streamlit app (entry point is `app.py`):

```bash
streamlit run app.py
```

You can also run individual Streamlit pages directly:

```bash
streamlit run pages/price_predictor.py
streamlit run pages/recommender_system.py
```

4. Open notebooks

Start Jupyter Lab / Notebook and open any `.ipynb` files for interactive exploration:

```bash
jupyter lab
# or
jupyter notebook
```

## Typical analysis workflow
1. Inspect raw data in `datasets/` with the EDA notebooks.
2. Run `missing_value.ipynb` to apply imputation strategies.
3. Run `outlier_treatment.ipynb` to identify and handle outliers.
4. Perform feature selection in `feature_selection.ipynb` and store cleaned datasets (`post_feature_selection*.csv`).
5. Train and evaluate models in `model_selection.ipynb` and `baseline_model.ipynb`.
6. Launch the Streamlit app to try predictions and visualizations interactively.

## Project contract (short)
- Inputs: CSV datasets found in `datasets/` (typical columns: location, area/size, price, features/amenities — inspect the notebooks for exact column names).
- Outputs: Trained models (artifacts kept in notebooks or exported files), visualization dashboards, and interactive predictions from Streamlit pages.
- Success criteria: cleaned data, reproducible notebook runs, Streamlit app launches, and reasonable model performance consistent with the experiments in `model_selection.ipynb`.

## Edge cases & notes
- Missing or inconsistent column names across CSVs — inspect and standardize before joining.
- Non-numeric inputs in numeric columns — feature engineering notebooks include encoders and conversions.
- Large CSVs: be mindful of memory limits; sample or use chunking when experimenting locally.
- No automated tests are included; add unit/tests when extracting code to modules.

## Development tips
- If you extract functionality from notebooks, move reusable functions into a package (e.g., `realestate/`) and add tests.
- Use `pip install -e .` while developing so Streamlit pages pick up code changes without reinstalling.

## Contributing
Contributions are welcome. Good first steps:
- Add an explicit `requirements.txt` or a lockfile for reproducible installs.
- Add a `LICENSE` file if you want to open-source this repository (MIT is a common choice).
- Add automated tests and small CI workflows for sanity checks.

## Where to look next
- `pyproject.toml` — dependency list and metadata.
- `app.py` and files in `pages/` — Streamlit app entry points.
- Notebooks — the canonical record of experiments and preprocessing steps.
