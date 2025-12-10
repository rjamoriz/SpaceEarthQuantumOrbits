#!/usr/bin/env python3
"""Add enhanced sections 1.3 and 4.5 to quantum_orbital_mathematics.ipynb"""
import json

# Read notebook
with open('quantum_orbital_mathematics.ipynb', 'r') as f:
    nb = json.load(f)

print(f"Original notebook: {len(nb['cells'])} cells")

# Find insertion points
section_2_idx = None
section_5_idx = None

for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'markdown':
        content = ''.join(cell['source'])
        if '## 2. Energy Components' in content:
            section_2_idx = i
        if '## 5. Quantum Optimization' in content:
            section_5_idx = i

print(f"Will insert section 1.3 at index {section_2_idx}")
print(f"Will insert section 4.5 at index {section_5_idx}")

# Section 1.3 cells (Real-World Orbital Perturbations)
section_1_3_cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "---\n",
            "## 1.3 Real-World Orbital Perturbations and Environmental Factors\n",
            "\n",
            "LEO satellites face numerous perturbations that must be considered in optimization.\n",
            "\n",
            "### 1.3.1 Atmospheric Drag in LEO\n",
            "\n",
            "**Drag Force:**\n",
            "$$\n",
            "\\vec{F}_{\\text{drag}} = -\\frac{1}{2} \\rho(h) C_D A v_{\\text{rel}}^2 \\hat{v}_{\\text{rel}}\n",
            "$$\n",
            "\n",
            "**Atmospheric Density Model:**\n",
            "$$\n",
            "\\rho(h) = \\rho_0 e^{-\\frac{h-h_0}{H}}\n",
            "$$\n",
            "\n",
            "Where $H \\approx 60$ km (scale height)\n",
            "\n",
            "**Orbital Lifetime:**\n",
            "- At 340 km: 1-2 years\n",
            "- At 550 km: 5-7 years\n",
            "- At 1200 km: 25+ years\n",
            "\n",
            "### 1.3.2 Thermal Environment\n",
            "\n",
            "**Temperature Range:**\n",
            "- Sunlit: +120°C to +150°C\n",
            "- Shadow: -100°C to -150°C\n",
            "- ΔT swing: ~250°C per orbit!\n",
            "\n",
            "**Heat Balance:**\n",
            "$$\n",
            "Q_{\\text{in}} = Q_{\\text{solar}} + Q_{\\text{albedo}} + Q_{\\text{Earth IR}}\n",
            "$$\n",
            "$$\n",
            "Q_{\\text{out}} = \\epsilon \\sigma A T^4\n",
            "$$\n",
            "\n",
            "### 1.3.3 Eclipse Analysis\n",
            "\n",
            "**Eclipse Fraction:**\n",
            "$$\n",
            "f_{\\text{eclipse}}(a) = \\frac{1}{\\pi}\\arccos\\left(\\sqrt{\\frac{a^2 - R_{\\oplus}^2}{a^2}}\\right)\n",
            "$$\n",
            "\n",
            "**Power Budget Constraint:**\n",
            "$$\n",
            "\\text{DoD} = \\frac{P_{\\text{load}} \\cdot t_{\\text{eclipse}}}{E_{\\text{battery}}} \\leq 0.3\n",
            "$$\n",
            "\n",
            "### 1.3.4 Space Debris Risk\n",
            "\n",
            "**Collision Probability:**\n",
            "$$\n",
            "P_{\\text{collision}} = n_{\\text{debris}}(h) \\cdot v_{\\text{rel}} \\cdot A_{\\text{cross}} \\cdot t\n",
            "$$\n",
            "\n",
            "LEO debris population:\n",
            "- Objects >10 cm: ~34,000\n",
            "- Objects >1 cm: ~1,000,000\n",
            "- Relative velocity: up to 15 km/s!\n",
            "\n",
            "**Kessler Syndrome:** Cascading collisions when:\n",
            "$$\n",
            "\\alpha N^2 > \\mu N\n",
            "$$\n",
            "\n",
            "### 1.3.5 Attitude Control\n",
            "\n",
            "**Disturbance Torques:**\n",
            "\n",
            "Gravity gradient:\n",
            "$$\n",
            "M_{\\text{gg}} = \\frac{3\\mu}{2r^3}|I_{\\text{max}} - I_{\\text{min}}|\\sin(2\\theta)\n",
            "$$\n",
            "\n",
            "Aerodynamic:\n",
            "$$\n",
            "M_{\\text{aero}} = \\frac{1}{2}\\rho v^2 A C_D d_{\\text{offset}}\n",
            "$$\n",
            "\n",
            "### 1.3.6 Propellant Budget\n",
            "\n",
            "**Total Delta-V:**\n",
            "$$\n",
            "\\Delta V_{\\text{total}} = \\Delta V_{\\text{deploy}} + \\Delta V_{\\text{maintain}} + \\Delta V_{\\text{deorbit}}\n",
            "$$\n",
            "\n",
            "**Station-Keeping (yearly):**\n",
            "- Low altitude (340 km): ~50 m/s/year\n",
            "- Medium altitude (550 km): ~10 m/s/year\n",
            "- High altitude (1200 km): ~2 m/s/year"
        ]
    }
]

# Section 4.5 cells (Enhanced QUBO with all perturbations)
section_4_5_cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "---\n",
            "### 4.5 Enhanced QUBO Formulation with Real-World Perturbations\n",
            "\n",
            "Integrating all orbital perturbations into the quantum optimization.\n",
            "\n",
            "#### 4.5.1 Complete Multi-Objective Hamiltonian\n",
            "\n",
            "$$\n",
            "\\boxed{\n",
            "\\begin{aligned}\n",
            "H_{\\text{complete}}(\\vec{x}) &= \\alpha_1 H_{\\text{orbital}} + \\alpha_2 H_{\\text{coverage}} + \\alpha_3 H_{\\text{collision}} \\\\\n",
            "&\\quad + \\alpha_4 H_{\\text{comm}} + \\alpha_5 H_{\\text{drag}} + \\alpha_6 H_{\\text{thermal}} \\\\\n",
            "&\\quad + \\alpha_7 H_{\\text{eclipse}} + \\alpha_8 H_{\\text{debris}} + \\alpha_9 H_{\\text{attitude}} \\\\\n",
            "&\\quad + \\alpha_{10} H_{\\text{propellant}} + P \\sum_c g_c(\\vec{x})^2\n",
            "\\end{aligned}\n",
            "}\n",
            "$$"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "#### 4.5.2 Atmospheric Drag Term\n",
            "\n",
            "$$\n",
            "H_{\\text{drag}}(\\vec{x}) = \\sum_{i=1}^{N} w_{\\text{drag}} \\cdot \\rho(h_i) \\cdot \\frac{A_i}{m_i} \\cdot v_{\\text{orb}}(a_i)\n",
            "$$\n",
            "\n",
            "**Binary Encoding:**\n",
            "$$\n",
            "h_i = h_{\\min} + \\sum_{k=0}^{n_h-1} 2^k \\Delta h \\cdot x_{i,k}^{(h)}\n",
            "$$\n",
            "\n",
            "**Lifetime Constraint:**\n",
            "$$\n",
            "t_{\\text{life}}(h) \\geq t_{\\text{mission}} + 5\\text{ years (deorbit margin)}\n",
            "$$"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "#### 4.5.3 Thermal Management Term\n",
            "\n",
            "$$\n",
            "H_{\\text{thermal}}(\\vec{x}) = \\sum_{i=1}^{N} w_{\\text{thermal}} \\cdot [\\Delta T_i(\\vec{x})]^2\n",
            "$$\n",
            "\n",
            "**Temperature Swing:**\n",
            "$$\n",
            "\\Delta T_i = T_{\\text{sun}}(h_i) - T_{\\text{eclipse}}(h_i)\n",
            "$$\n",
            "\n",
            "Lower altitudes → higher drag → more heating"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "#### 4.5.4 Eclipse and Power Term\n",
            "\n",
            "$$\n",
            "H_{\\text{eclipse}}(\\vec{x}) = \\sum_{i=1}^{N} w_{\\text{eclipse}} \\cdot \\left(\\frac{P_{\\text{load}} \\cdot t_{\\text{eclipse}}(a_i)}{0.3 \\cdot E_{\\text{battery}}}\\right)^2\n",
            "$$\n",
            "\n",
            "**Eclipse Duration:**\n",
            "$$\n",
            "t_{\\text{eclipse}}(a) = \\frac{T_{\\text{orb}}(a)}{\\pi}\\arccos\\left(\\frac{R_{\\oplus}}{a}\\right)\n",
            "$$"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "#### 4.5.5 Space Debris Risk Term\n",
            "\n",
            "$$\n",
            "H_{\\text{debris}}(\\vec{x}) = \\sum_{i=1}^{N} w_{\\text{debris}} \\cdot n_{\\text{debris}}(h_i) \\cdot A_{\\text{cross}} \\cdot v_{\\text{rel}}(i_i)\n",
            "$$\n",
            "\n",
            "**Debris Density Zones:**\n",
            "$$\n",
            "n_{\\text{debris}}(h) = \\begin{cases}\n",
            "n_{\\text{high}} & h \\in [340, 600] \\text{ km} \\\\\n",
            "n_{\\text{med}} & h \\in [600, 900] \\text{ km} \\\\\n",
            "n_{\\text{low}} & h \\in [900, 1200] \\text{ km}\n",
            "\\end{cases}\n",
            "$$"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "#### 4.5.6 Attitude Control Term\n",
            "\n",
            "$$\n",
            "H_{\\text{attitude}}(\\vec{x}) = \\sum_{i=1}^{N} w_{\\text{att}} \\cdot |M_{\\text{dist}}(a_i, h_i)|\n",
            "$$\n",
            "\n",
            "Lower altitude → stronger magnetic field → better control authority"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "#### 4.5.7 Propellant Budget Term\n",
            "\n",
            "$$\n",
            "H_{\\text{propellant}}(\\vec{x}) = \\sum_{i=1}^{N} w_{\\Delta V} \\cdot \\Delta V_{\\text{total}}(\\vec{x}_i)\n",
            "$$\n",
            "\n",
            "**Components:**\n",
            "- Deploy: Hohmann transfer from parking orbit\n",
            "- Maintain: Drag compensation (altitude-dependent)\n",
            "- Deorbit: 25-year compliance"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "#### 4.5.8 Final Comprehensive QUBO\n",
            "\n",
            "$$\n",
            "\\boxed{\n",
            "\\begin{aligned}\n",
            "H_{\\text{Starlink}}(\\vec{x}) &= \\alpha_1 \\sum_{i} \\frac{\\mu}{2a_i} - \\alpha_2 \\sum_{i,j} V_{ij} \\cdot \\text{vis}(i,j) \\\\\n",
            "&\\quad + \\alpha_3 \\sum_{i<j} \\frac{K}{d_{ij}^2} - \\alpha_4 \\sum_{i,j} \\frac{P_{\\text{link}}}{d_{ij}^2} \\\\\n",
            "&\\quad + \\alpha_5 \\sum_{i} \\frac{\\rho(h_i) A_i v_i}{m_i} + \\alpha_6 \\sum_{i} \\Delta T_i^2 \\\\\n",
            "&\\quad + \\alpha_7 \\sum_{i} \\left(\\frac{P t_{\\text{eclipse}}}{0.3 E}\\right)^2 + \\alpha_8 \\sum_{i} n_{\\text{debris}}(h_i) \\\\\n",
            "&\\quad + \\alpha_9 \\sum_{i} M_{\\text{dist}}(h_i) + \\alpha_{10} \\sum_{i} \\Delta V_{\\text{total},i} \\\\\n",
            "&\\quad + P \\sum_c g_c(\\vec{x})^2\n",
            "\\end{aligned}\n",
            "}\n",
            "$$\n",
            "\n",
            "**Problem Complexity:**\n",
            "- **Class**: NP-Hard\n",
            "- **Objectives**: 10 competing goals\n",
            "- **Variables**: 1000-10000 binary\n",
            "- **Search space**: $2^{1000}$ to $2^{10000}$ configurations\n",
            "- **Real-world physics**: Drag, thermal, debris, eclipses, J2\n",
            "\n",
            "**Quantum Advantage:** Essential for finding optimal solutions!"
        ]
    }
]

# Insert sections
if section_2_idx:
    for idx, cell in enumerate(section_1_3_cells):
        nb['cells'].insert(section_2_idx + idx, cell)
    print(f"✓ Inserted section 1.3 ({len(section_1_3_cells)} cells)")
    
    # Adjust section_5_idx since we inserted cells
    section_5_idx += len(section_1_3_cells)

if section_5_idx:
    for idx, cell in enumerate(section_4_5_cells):
        nb['cells'].insert(section_5_idx + idx, cell)
    print(f"✓ Inserted section 4.5 ({len(section_4_5_cells)} cells)")

# Save
with open('quantum_orbital_mathematics.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)

print(f"\n✓ Enhanced notebook now has {len(nb['cells'])} cells")
print("✓ Added comprehensive real-world orbital mechanics formulations")
