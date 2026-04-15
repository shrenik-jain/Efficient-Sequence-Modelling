# Efficient sequence modelling: KV cache — MHA vs GQA

This repository supports a UCSD ECE 226 project on **KV-cache memory footprint** during long-context inference, comparing **multi-head attention (MHA)** with **grouped-query attention (GQA)**.

## Why this matters

During autoregressive generation, past keys and values are often stored in a **KV cache** whose size grows linearly with sequence length. GQA uses fewer key/value heads than query heads, so the cache can be much smaller than full MHA at the same model width—an important lever for long-context deployment.

## Repository contents

| Item | Description |
|------|-------------|
| [`MHA vs GQA.ipynb`](MHA%20vs%20GQA.ipynb) | Full notebook: synthetic Llama-shaped models, pretrained comparisons, KV sweeps, and an optional long-context **needle-in-a-haystack (MCQ)** benchmark with quantized-cache experiments. |
| [`kv_cache_mha_gqa.py`](kv_cache_mha_gqa.py) | Standalone script with the same core KV-growth measurements (synthetic and optional pretrained runs), suitable for terminals and automation. |

## Setup

Use Python 3.10+ and a virtual environment.

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Install PyTorch for your platform (see https://pytorch.org/get-started/locally/)
pip install torch

pip install -r requirements.txt
```

**GPU:** A CUDA-capable GPU is strongly recommended. The pretrained path loads Hugging Face models with `device_map="auto"` and is oriented toward GPU memory.

**Headless plotting:** To save figures without a display, set `MPLBACKEND=Agg` (the script saves PNGs when `--output-dir` is set).

## Run the script

### Synthetic models (no downloads)

Builds randomly initialized `LlamaForCausalLM` configs and measures empirical vs analytical KV size as the cache grows.

```bash
python kv_cache_mha_gqa.py synthetic --output-dir ./figures
```

Useful flags: `--total-tokens`, `--gqa-kv-heads`, `--device cpu|cuda`, `--show` (open plot windows).

### Pretrained models (downloads weights)

Compares two Hugging Face causal LMs (defaults match the notebook: OpenLLaMA 3B MHA vs TinyLlama 1.1B Chat GQA).

```bash
python kv_cache_mha_gqa.py pretrained --output-dir ./figures
```

Override models if needed:

```bash
python kv_cache_mha_gqa.py pretrained \
  --mha-model openlm-research/open_llama_3b \
  --gqa-model TinyLlama/TinyLlama-1.1B-Chat-v1.0
```

## Notebook map

1. **Install & setup** — dependencies and device checks.  
2. **Synthetic Llama** — MHA vs GQA KV growth, analytical cross-check, optional sweep over `num_key_value_heads`.  
3. **Pretrained models** — same measurement on real checkpoints.  
4. **Needle in a haystack (MCQ)** — long-context retrieval stress test; optional HQQ/Quanto quantized KV cache modes when those libraries are installed.

## Results interpretation

- **Empirical KV** is computed from the model’s `past_key_values` tensors (byte-accurate for the returned cache structure).  
- **Analytical KV** uses \(2 \times L \times B \times T \times H_{\text{kv}} \times D_h\) bytes (keys + values), matching standard GQA/MHA sizing assumptions.  
- The ratio of MHA to GQA KV at the same sequence length often tracks the ratio of `num_heads` to `num_key_value_heads` when head dimension and layer count are held fixed (see script output “ideal ~Nx”).

## License

See [LICENSE](LICENSE).
