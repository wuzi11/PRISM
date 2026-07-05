# PRISM

LLM pathway-prior conditioned perturbation model for single-cell drug response prediction.

## Setup

```bash
pip install -r requirements.txt
```

Copy and edit `config/llm.env` before building LLM priors:

```bash
cp config/llm.env.example config/llm.env
```

```
OPENAI_API_KEY=your_key
OPENAI_BASE_URL=your_api_base
LLM_MODEL=your_model_name
```

## Data

Place `.h5ad` files under `datasets/` (sciplex) or `datasets_lincs/` (lincs).  
Each file must contain `obs["Group"]` in the form `{cell_line}_{drug}_{dose}`.

Default sciplex layout:

```
datasets/sci_plex_train_drug_split_0.h5ad
datasets/sci_plex_train_drug_split_0_control.h5ad
datasets/sci_plex_test_drug_split_0.h5ad
datasets/sci_plex_test_drug_split_0_control.h5ad
```

Or pass absolute paths via `DATA_PATH`, `CONTROL_DATA_PATH`, `TEST_ADATA_PATH`, `TEST_CONTROL_ADATA_PATH`.

## Pre-built priors

```
resources/deepseek/llm_pathway_priors.json
resources/qwen/llm_pathway_priors.json
resources/llm_pathway_priors_sci_plex_zero.json
resources/llm_pathway_priors_sci_plex_random.json
```

## Run pipeline

From the repository root:

```bash
# sciplex + deepseek priors (default)
bash shell/run_prism.sh

# sciplex + qwen priors
DATASET=sciplex PRIOR=qwen bash shell/run_prism.sh

# zero / random priors
PRIOR=zero bash shell/run_prism.sh

# lincs
DATASET=lincs SPLIT=drug_split_1 PRIOR=zero bash shell/run_prism.sh
```

Build priors only:

```bash
DATA_PATH=datasets/sci_plex_train_drug_split_0.h5ad \
TEST_ADATA_PATH=datasets/sci_plex_test_drug_split_0.h5ad \
PRIOR_PATH=resources/deepseek/llm_pathway_priors_sci_plex.json \
bash shell/build_llm_priors.sh
```

Train / infer directly:

```bash
python scripts/train_prism.py --help
python scripts/infer_prism.py --help
```

## Outputs

| Step | Location |
|------|----------|
| Checkpoints | `checkpoints/<run_name>/model.pt` |
| Predictions | `results/<run_name>/predictions/perturbed_expression_prism.npy` |
| Logs | `logs/logger_files/` |

## Options

| Variable | Values | Default |
|----------|--------|---------|
| `DATASET` | sciplex, lincs | sciplex |
| `SPLIT` | drug_split_0, ... | drug_split_0 |
| `PRIOR` | deepseek, qwen, zero, random | deepseek |
| `BUILD_PRIORS` | auto, true, false | auto |
