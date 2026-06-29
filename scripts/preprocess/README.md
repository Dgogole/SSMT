# Remeshing Script Documentation

## Overview

`remesh.sh` is a preprocessing script for 3D dental mesh data. It converts raw OBJ models into a standardized mesh representation in three steps.

## Pipeline

### Step 1: Manifold Processing (`manifold.py`)

**Purpose:** Convert each input mesh into a manifold mesh and repair topology issues.

**Input:**

- Directory: `$DATAROOT/single_before` (contains the original `.obj` files)
- Format: 3D meshes in OBJ format

**Output:**

- Directory: `$MANIFOLD/manifold_before`
- Format: repaired `.obj` files

**Processing details:**

- Uses the `Manifold/build/manifold` executable
- Target face count = original face count × 1.2
- Repairs topology defects such as non-manifold edges and self-intersections
- Processes files in parallel with 16 workers

**Required tool:**

```bash
/home/dtl/workspace/TADPM/Manifold/build/manifold <input> <output> <face_count>
```

---

### Step 2: Simplification (`simplify.py`)

**Purpose:** Simplify each manifold mesh to a fixed face count of 256.

**Input:**

- Directory: `$MANIFOLD/manifold_before` (output of Step 1)

**Output:**

- Directory: `$SIMPLIFY/simplify_before`
- Format: simplified `.obj` files with exactly 256 faces

**Processing details:**

- Uses the `pyfqmr` (Fast Quadric Mesh Simplification) library
- Target face count: 256
- Parameters:
  - `aggressiveness=7`: simplification strength
  - `preserve_border=True`: preserves mesh boundaries
- Processes files in parallel with 16 workers
- Automatically skips files that already exist

---

### Step 3: MAPS Remeshing (`datagen_maps.py`)

**Purpose:** Apply multiresolution adaptive remeshing with the MAPS algorithm.

**Input:**

- Directory: `$SIMPLIFY/simplify_before` (output of Step 2)

**Output:**

- Directory: `$REMESH/remesh_before`
- Format: remeshed `.obj` files
- Final face count: 256 × 64 = 16,384

**Processing details:**

- Uses MAPS (Multiresolution Adaptive Parameterization of Surfaces)
- Parameters:
  - `base_size=256`: base mesh face count
  - `depth=3`: upsampling depth (2^3 = 8×)
  - `timeout=20`: timeout for each file
- Processes files in parallel with 32 workers
- Displays progress with `tqdm`
- Automatically skips files that already exist

**Reference:**

> Lee, Aaron W. F., et al. "MAPS: Multiresolution adaptive parameterization of surfaces."
> Proceedings of the 25th Annual Conference on Computer Graphics and Interactive Techniques, 1998.

---

## Input and Output Summary

### Input Requirements

- **Format:** OBJ
- **Location:** Files must be placed directly in the input directory; subdirectories are not supported
- **Naming:** Any filename is accepted and preserved in the output
- **Mesh requirements:** Valid triangular meshes

### Output Directory Structure

```text
$data_root/
├── single_before/         # Original input provided by the user
├── manifold_before/       # Step 1 output: topology repair
├── simplify_before/       # Step 2 output: simplified to 256 faces
└── remesh_before/         # Step 3 output: MAPS remeshing
```

### Mesh Specifications by Stage

| Stage | Face count | Description |
| --- | ---: | --- |
| Original input | Varies (test sample: 5,790) | Original user data |
| Manifold | Original × 1.2 (test sample: ~6,948) | Topology repaired |
| Simplify | 256 | Fixed-size simplified mesh |
| Remesh | 16,384 (256 × 64) | MAPS upsampling |

---

## Dependencies

### Python Libraries

```text
trimesh      # Mesh loading and processing
vedo         # Mesh loading for manifold.py
pyfqmr       # Fast quadric mesh simplification
numpy        # Numerical computing
tqdm         # Progress display
```

### C++ Executable

```bash
/home/dtl/workspace/TADPM/Manifold/build/manifold  # Manifold repair
# simplify.py uses pyfqmr and no longer depends on the C++ simplifier.
```

---

## User Data Compatibility Check

**Data path:** `/data/dtl/multimodal_teeth_dataset`

**Results:**

- The files use the correct OBJ format
- The files are placed directly in the `single_before/` directory
- The naming scheme is compatible: `P0001_11.obj` (`PID_ToothID`)
- The meshes are valid and can be loaded by `trimesh`
  - Example: `P0001_11.obj` contains 2,897 vertices and 5,790 faces

**Conclusion:** The data is fully compatible with `remesh.sh`; no changes are required.

---

## Usage

### Basic Usage

```bash
cd /home/dtl/workspace/TADPM/scripts
bash remesh.sh
```

### Change the Data Paths

To process a different dataset, edit `remesh.sh`:

```bash
ROOT=/path/to/your/data
DATAROOT=$ROOT/single_before      # Original data directory
MANIFOLD=$ROOT/manifold_before    # Intermediate output directory
SIMPLIFY=$ROOT/simplify_before    # Intermediate output directory
REMESH=$ROOT/remesh_before        # Final output directory
```

### Run Individual Steps

```bash
# Run Step 1 only
python data_preprocess/manifold.py \
    --dataroot "$DATAROOT" \
    --manifold "$MANIFOLD" \
    --simplify "$SIMPLIFY"

# Run Step 2 only
python data_preprocess/simplify.py \
    --dataroot "$DATAROOT" \
    --manifold "$MANIFOLD" \
    --simplify "$SIMPLIFY"

# Run Step 3 only
python data_preprocess/datagen_maps.py \
    --simplify "$SIMPLIFY" \
    --output "$REMESH"
```

---

## Notes

1. **Reusing intermediate files:** Each script checks whether its output already exists and skips existing files, so interrupted runs can be resumed safely.

2. **Parallel processing:**
   - `manifold.py`: 16 workers
   - `simplify.py`: 16 workers
   - `datagen_maps.py`: 32 workers

3. **Memory usage:** Parallel processing can consume substantial memory. Reduce the worker count if the process runs out of memory.

4. **Storage:** Allow approximately three to four times the size of the original data for intermediate outputs.

5. **Timeout handling:** The MAPS step applies a 60-second timeout to each file. Timed-out files are skipped and reported as errors.

---

## Troubleshooting

### Issue 1: The Manifold Executable Is Missing

```bash
cd /home/dtl/workspace/TADPM/Manifold
mkdir -p build && cd build
cmake ..
make
```

### Issue 2: Python Dependencies Are Missing

```bash
pip install trimesh vedo pyfqmr numpy tqdm
```

### Issue 3: Some Files Fail to Process

- Verify that the original mesh is valid by loading it with `trimesh`
- Check the error log for the affected filename
- Check file permissions and available disk space

---

## File Locations

- Script: `/home/dtl/workspace/TADPM/scripts/remesh.sh`
- Python modules: `/home/dtl/workspace/TADPM/data_preprocess/`
- Manifold tool: `/home/dtl/workspace/TADPM/Manifold/`
