#!/usr/bin/env python3
"""
Organize and rename visualization files with descriptive names.

This script copies all CAZ visualization PNGs from timestamped folders
into a single organized directory with concept-model-date naming.
"""

import json
import shutil
from pathlib import Path
from datetime import datetime

# Map folder patterns to concepts
CONCEPT_MAPPING = {
    "caz_validation": "credibility",
    "negation": "negation",
    "sentiment": "sentiment",
}

def get_concept_from_folder(folder_name):
    """Extract concept name from folder."""
    for pattern, concept in CONCEPT_MAPPING.items():
        if pattern in folder_name:
            return concept
    return "unknown"

def get_model_from_analysis(analysis_file):
    """Extract model ID from analysis JSON."""
    try:
        with open(analysis_file) as f:
            data = json.load(f)
            return data.get('model_id', 'unknown')
    except:
        return 'unknown'

def organize_visualizations():
    """Reorganize all visualizations with descriptive names."""
    results_dir = Path("Rosetta_Manifold/results")
    viz_output_dir = Path("Rosetta_Manifold/visualizations")
    viz_output_dir.mkdir(exist_ok=True)

    organized_files = []

    # Scan all result directories
    for result_dir in results_dir.iterdir():
        if not result_dir.is_dir():
            continue

        # Look for visualization files
        viz_files = list(result_dir.glob("caz_visualization_*.png"))
        if not viz_files:
            continue

        # Get concept from folder name
        concept = get_concept_from_folder(result_dir.name)

        # Get timestamp from folder name (if exists)
        timestamp = "unknown"
        parts = result_dir.name.split("_")
        for part in parts:
            if len(part) == 8 and part.isdigit():  # YYYYMMDD
                try:
                    dt = datetime.strptime(part, "%Y%m%d")
                    timestamp = dt.strftime("%Y-%m-%d")
                except:
                    pass

        for viz_file in viz_files:
            # Extract model from filename
            model = viz_file.stem.replace("caz_visualization_", "")

            # Try to get more accurate model from analysis file
            analysis_files = list(result_dir.glob("caz_analysis_*.json"))
            if analysis_files:
                model = get_model_from_analysis(analysis_files[0])

            # Create descriptive filename
            new_name = f"{concept}_{model}_{timestamp}.png"
            new_path = viz_output_dir / new_name

            # Copy file
            shutil.copy2(viz_file, new_path)
            print(f"Copied: {viz_file.name} → {new_name}")

            organized_files.append({
                'concept': concept,
                'model': model,
                'date': timestamp,
                'original': str(viz_file),
                'organized': str(new_path)
            })

    # Create an index file
    index_md = viz_output_dir / "INDEX.md"
    with open(index_md, 'w') as f:
        f.write("# Rosetta Manifold - Visualization Index\n\n")
        f.write("All CAZ (Concept Assembly Zone) visualizations organized by concept and model.\n\n")

        # Group by concept
        by_concept = {}
        for item in organized_files:
            concept = item['concept']
            if concept not in by_concept:
                by_concept[concept] = []
            by_concept[concept].append(item)

        for concept in sorted(by_concept.keys()):
            f.write(f"## {concept.title()}\n\n")
            f.write("| Model | Date | File |\n")
            f.write("|:------|:-----|:-----|\n")

            items = sorted(by_concept[concept], key=lambda x: (x['model'], x['date']))
            for item in items:
                f.write(f"| {item['model']} | {item['date']} | `{Path(item['organized']).name}` |\n")
            f.write("\n")

        f.write("## All Visualizations\n\n")
        f.write("| Concept | Model | Date | Filename |\n")
        f.write("|:--------|:------|:-----|:---------|\n")
        for item in sorted(organized_files, key=lambda x: (x['concept'], x['model'], x['date'])):
            filename = Path(item['organized']).name
            f.write(f"| {item['concept']} | {item['model']} | {item['date']} | `{filename}` |\n")

    print(f"\n✅ Organized {len(organized_files)} visualizations")
    print(f"📁 Output directory: {viz_output_dir}")
    print(f"📋 Index created: {index_md}")

if __name__ == "__main__":
    organize_visualizations()
