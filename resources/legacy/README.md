# Legacy / deprecated notebooks

These notebooks are **prototype artifacts** and are **not** part of the
reported experimental pipeline. They are kept only for historical reference.

- `remote_gemma.ipynb` — early Gemma prototype (HF `AutoModelForCausalLM`, no vLLM). Superseded.
- `kaggle_run.ipynb`, `colab_run.ipynb` — early Kaggle/Colab launch helpers. Superseded.

**Official notebooks for the reported results** live in `resources/`:

- `tests_ptx.ipynb` — end-to-end pipeline (Qwen3.5-9B-AWQ served with vLLM on an L4).
- `tests_raw_llm.ipynb` — zero-shot raw-LLM baseline.
