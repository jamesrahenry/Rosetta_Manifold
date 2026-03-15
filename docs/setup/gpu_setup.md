# GPU Setup — PyTorch CUDA on Systems Without a System CUDA Toolkit

This covers the case where `nvidia-smi` shows your GPU but `torch.cuda.is_available()` returns `False`. Common on Ubuntu systems where the NVIDIA driver is installed but the CUDA toolkit is not, including WSL2 environments.

Tested on: Ubuntu 22.04 / WSL2, NVIDIA RTX 500 Ada (4GB), driver 575.x (CUDA 13.0 capability), Python 3.11 via pyenv.

---

## The Problem

`nvidia-smi` works — the driver sees the GPU. But PyTorch can't use it because the CUDA runtime libraries (`libcudart`, `libcublas`, etc.) aren't on the system. Installing the system CUDA toolkit is one fix; installing them via pip is simpler and doesn't require root.

---

## Step 1: Install PyTorch with CUDA support

The standard `pip install torch` gives you the CPU-only build. Force-reinstall with the CUDA 12.4 index:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124 --force-reinstall
```

This pulls the CUDA-enabled wheel. However, it may not bundle all required runtime libraries depending on your pip version and existing environment.

## Step 2: Install the nvidia CUDA runtime packages

PyTorch's CUDA wheel depends on nvidia's pip-packaged runtime libraries. Install them explicitly:

```bash
pip install \
    nvidia-cuda-runtime-cu12 \
    nvidia-cuda-nvrtc-cu12 \
    nvidia-cublas-cu12 \
    nvidia-cudnn-cu12
```

If you get version conflicts, let pip resolve them — the torch cu124 wheel pins specific versions of these packages and pip will find the right ones.

## Step 3: Verify

```bash
python3 -c "
import torch
print('CUDA available:', torch.cuda.is_available())
print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')
print('VRAM:', round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1), 'GB')
"
```

Expected output on a 4GB card:
```
CUDA available: True
GPU: NVIDIA RTX 500 Ada Generation Laptop GPU
VRAM: 4.0 GB
```

---

## Running GPT-2-XL on 4GB VRAM

GPT-2-XL has 1.5B parameters. In fp32 that's ~6GB — won't fit. In fp16 it's ~3GB — fits with room for activations.

Load with `torch_dtype=torch.float16`:

```python
from transformers import GPT2Model, GPT2LMHeadModel, GPT2Tokenizer
import torch

device = torch.device("cuda")
dtype  = torch.float16

# For hidden state analysis (no language model head)
model = GPT2Model.from_pretrained("gpt2-xl", torch_dtype=dtype).to(device)

# For generation
model = GPT2LMHeadModel.from_pretrained("gpt2-xl", torch_dtype=dtype).to(device)
```

Check VRAM after loading:
```python
used  = torch.cuda.memory_allocated() / 1024**3
total = torch.cuda.get_device_properties(0).total_memory / 1024**3
print(f"VRAM: {used:.2f}/{total:.1f} GB")
# Expected: ~3.0/4.0 GB
```

**fp16 caveat:** cosine similarity and other floating-point comparisons should cast to fp32 first to avoid precision loss:
```python
def cosine_sim(a, b):
    a, b = a.float(), b.float()
    return torch.dot(a, b) / (torch.norm(a) * torch.norm(b) + 1e-8)
```

---

## TransformerLens on 4GB

TransformerLens loads models differently. Use `dtype="float16"` and move to device:

```python
import transformer_lens
model = transformer_lens.HookedTransformer.from_pretrained(
    "gpt2-xl",
    dtype="float16",
).to("cuda")
```

Note: some TransformerLens operations internally upcast to fp32. If you hit OOM during a hook-heavy forward pass, reduce batch size to 1 or move to CPU for that operation.

---

## Ollama CUDA Libraries (alternative path)

If Ollama is installed, its bundled CUDA libraries are at `/usr/local/lib/ollama/cuda_v12/`. You can point the linker there as a fallback:

```bash
export LD_LIBRARY_PATH=/usr/local/lib/ollama/cuda_v12:/usr/lib/wsl/lib:$LD_LIBRARY_PATH
```

This works for some setups but is fragile — the pip approach above is more reliable.

---

## SSH Config for GitHub (personal key)

If using multiple GitHub accounts, the SSH config host alias pattern avoids credential conflicts:

```
# ~/.ssh/config
Host github-personal
    HostName github.com
    User git
    IdentityFile ~/.ssh/your-personal-key
    PreferredAuthentications publickey
    IdentitiesOnly yes
```

Then use `git remote set-url origin git@github-personal:username/repo.git` instead of the standard `git@github.com:...` form.

---

## TransformerLens + Pythia on transformers 5.x

TransformerLens 2.16.x reads `hf_config.rotary_pct` directly when loading
GPTNeoX (Pythia) models. transformers 5.x moved this attribute into
`rope_parameters.partial_rotary_factor`, causing:

```
AttributeError: 'GPTNeoXConfig' object has no attribute 'rotary_pct'
```

**Fix:** patch the installed TransformerLens package in-place:

```bash
TLLM_FILE=$(python3 -c "import transformer_lens, os; print(os.path.join(os.path.dirname(transformer_lens.__file__), 'loading_from_pretrained.py'))")

sed -i 's/        rotary_pct = hf_config\.rotary_pct/        rotary_pct = getattr(hf_config, "rotary_pct", None) or (hf_config.rope_parameters.get("partial_rotary_factor", 1.0) if hasattr(hf_config, "rope_parameters") else 1.0)/' $TLLM_FILE
```

This falls back to `rope_parameters.partial_rotary_factor` (0.25 for Pythia)
when `rotary_pct` is absent. The fix survives until `transformer-lens` is
reinstalled or upgraded.

Alternatively, upgrade to TransformerLens 2.17.0 which may fix this natively:
```bash
pip install transformer-lens==2.17.0
```
(Untested as of 2026-03-15 — verify pythia loads before running long jobs.)
