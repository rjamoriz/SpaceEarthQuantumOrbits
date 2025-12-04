#!/usr/bin/env python3
"""
Add header image to the beginning of the Jupyter notebook.
"""

import json
import sys

notebook_path = "starlink_3d_quantum_optimization.ipynb"

print(f"📝 Adding header image to {notebook_path}...")

# Read the notebook
with open(notebook_path, 'r', encoding='utf-8') as f:
    notebook = json.load(f)

# Create new markdown cell with image
header_cell = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "# 🌍 Starlink 3D Quantum Optimization\n",
        "\n",
        "![Starlink Constellation](starlink_reference_image.png)\n",
        "\n",
        "*Professional visualization of the Starlink satellite constellation - our goal is to create similar photorealistic renders.*\n",
        "\n",
        "---\n"
    ]
}

# Insert at position 1 (after the title cell)
notebook['cells'].insert(1, header_cell)

# Save the modified notebook
with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1)

print("✅ Header image cell added successfully!")
print("📍 Position: Cell #1 (after title)")
print("\n💡 Note: Make sure 'starlink_reference_image.png' is in the same directory")
print("   You can save the reference image manually to the project folder.")
