# Visualization Improvements

## Summary of Changes

All CAZ (Concept Assembly Zone) visualizations now include the concept name in both the filename and the plot title, making them self-documenting and shareable without relying on folder context.

## What Changed

### 1. Updated Code (`src/analyze_caz.py`)

**Before**:
- Filename: `caz_visualization_{model}.png` (e.g., `caz_visualization_gpt2.png`)
- Title: `Concept Assembly Zone: {model}` (e.g., `Concept Assembly Zone: gpt2`)

**After**:
- Filename: `caz_visualization_{concept}_{model}.png` (e.g., `caz_visualization_credibility_gpt2.png`)
- Title: `Concept Assembly Zone: {Concept} ({model})` (e.g., `Concept Assembly Zone: Credibility (gpt2)`)

**New Parameter**:
```bash
python src/analyze_caz.py \
    --input results/extraction.json \
    --output-dir results/ \
    --concept credibility  # ← NEW!
```

### 2. Updated Run Scripts

All three concept-specific run scripts now pass the concept name:

- `run_caz_validation.sh` - Updated to accept concept as second argument
- `run_sentiment_suite.sh` - Passes `--concept sentiment`
- `run_negation_suite.sh` - Passes `--concept negation`

### 3. Organized Existing Visualizations

Created `Rosetta_Manifold/visualizations/` directory with all visualizations renamed descriptively:

```
visualizations/
├── credibility_gpt2_2026-03-10.png
├── credibility_gpt2-xl_2026-03-10.png
├── negation_gpt2_2026-03-10.png
├── negation_gpt2-xl_2026-03-10.png
├── sentiment_gpt2_2026-03-10.png
├── sentiment_gpt2-xl_2026-03-10.png
└── INDEX.md  # ← Full listing with metadata
```

## Benefits

### Before
❌ **Folder-dependent context**:
```
results/caz_validation_gpt2_20260310_164336/
  └── caz_visualization_gpt2.png  # What concept is this?
results/negation_gpt2_20260310_210541/
  └── caz_visualization_gpt2.png  # Same name, different concept!
```

### After
✅ **Self-documenting filenames**:
```
visualizations/
  ├── credibility_gpt2_2026-03-10.png  # Clear!
  └── negation_gpt2_2026-03-10.png     # Clear!
```

✅ **Clear plot titles**:
- "Concept Assembly Zone: Credibility (gpt2)"
- "Concept Assembly Zone: Negation (gpt2-xl)"
- "Concept Assembly Zone: Sentiment (gpt2)"

## Usage Examples

### Running with New Scripts

**Credibility** (default):
```bash
./run_caz_validation.sh gpt2 credibility
```

**Negation**:
```bash
./run_negation_suite.sh
# Automatically passes --concept negation
```

**Sentiment**:
```bash
./run_sentiment_suite.sh
# Automatically passes --concept sentiment
```

**Custom concept**:
```bash
# 1. Generate your dataset (e.g., honesty_pairs.jsonl)
# 2. Extract metrics
python src/extract_vectors_caz.py \
    --model gpt2 \
    --dataset data/honesty_pairs.jsonl \
    --output results/honesty_extraction.json

# 3. Analyze with concept name
python src/analyze_caz.py \
    --input results/honesty_extraction.json \
    --output-dir results/ \
    --concept honesty  # ← Includes in filename and title!
```

### Organizing Existing Visualizations

To re-organize all existing visualizations:
```bash
python organize_visualizations.py
```

This creates:
- Copied files in `visualizations/` with descriptive names
- `INDEX.md` with full listing by concept

## Backward Compatibility

✅ **Fully backward compatible**:
- Default `--concept` value is `"concept"` if not specified
- Old scripts still work (just without concept in title/filename)
- No changes to data formats or analysis logic

## Scaling to 100+ Concepts

With these improvements, when you scale to many concepts:

### Organized Storage
```
visualizations/
├── credibility_gpt2_2026-03-10.png
├── credibility_llama3-8b_2026-03-15.png
├── honesty_gpt2_2026-03-11.png
├── honesty_mistral-7b_2026-03-15.png
├── bias_gpt2_2026-03-12.png
├── ... (100+ files with clear names)
└── INDEX.md
```

### Easy Sharing
When you send a visualization to a colleague:
- ✅ Filename tells them what it is: `credibility_gpt2-xl_2026-03-10.png`
- ✅ Title in the plot confirms it: "Concept Assembly Zone: Credibility (gpt2-xl)"
- ❌ No confusion about folder context

### Programmatic Access
```python
import glob

# Find all credibility visualizations
credibility_plots = glob.glob("visualizations/credibility_*.png")

# Find all gpt2-xl visualizations
xl_plots = glob.glob("visualizations/*_gpt2-xl_*.png")

# Find specific concept-model combination
plot = "visualizations/sentiment_gpt2_2026-03-10.png"
```

## Visualization Index

See `visualizations/INDEX.md` for a complete organized listing of all visualizations grouped by:
- Concept (credibility, negation, sentiment)
- Model (gpt2, gpt2-xl)
- Date

---

## Summary

**What you asked for**: ✅ Graphs well labelled with the concept being analyzed, shareable without folder context

**What we delivered**:
1. ✅ Concept name in filename: `caz_visualization_{concept}_{model}.png`
2. ✅ Concept name in plot title: `Concept Assembly Zone: {Concept} ({model})`
3. ✅ All run scripts updated to pass concept parameter
4. ✅ Existing visualizations organized in `visualizations/` directory
5. ✅ INDEX.md for easy browsing
6. ✅ Fully scalable to 100+ concepts

**Next time you run**:
```bash
./run_sentiment_suite.sh
# Creates: caz_visualization_sentiment_gpt2.png  ← Clear!
# Title: "Concept Assembly Zone: Sentiment (gpt2)"  ← Clear!
```

🎨 **All future visualizations will be self-documenting and easily shareable!**
