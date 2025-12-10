import json

with open('quantum_orbital_mathematics.ipynb', 'r') as f:
    nb = json.load(f)

print(f"Total cells: {len(nb['cells'])}")

# Check for the new section
found_qubo_enhanced = False
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'markdown':
        content = ''.join(cell['source'])
        if '4.5 Enhanced QUBO Formulation' in content:
            found_qubo_enhanced = True
            print(f"Found enhanced QUBO section at cell {i}")
            break

if found_qubo_enhanced:
    print("✓ Notebook successfully updated with comprehensive QUBO formulation!")
else:
    print("✗ Update may not have completed")
