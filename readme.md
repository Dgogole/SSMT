# SSMT

## Installation

```shell
conda create --name ssmt python=3.10
conda activate ssmt
pip install -r requirements.txt
```

Install chamfer distance:

```shell
cd chamfer_dist
python setup.py install
```

For `manifold` binary, the original repo is no longer available. Please contact me.

## 1. Data Preprocessing

The preprocessing pipeline converts raw dental scans into the formats needed for training: single tooth meshes, point clouds, remeshed surfaces, and rotation parameters.

### 1.1 Extract Single Tooth Meshes

```shell
bash scripts/preprocess/get_mesh.sh
```

Extracts individual tooth meshes from full dental models. Automatically centers and normalizes meshes.

### 1.2 Generate Point Clouds

```shell
bash scripts/preprocess/get_pointcloud.sh
```

Extracts corresponding point clouds from individual teeth before and after orthodontic treatment.

### 1.3 Remesh

```shell
bash scripts/preprocess/remesh.sh
```

Three-step pipeline: Manifold processing, Simplify processing, then data generation (MAPS). Converts raw meshes into remeshed surfaces suitable for MeshMAE encoding.

### 1.4 Compute Rotation Parameters

```shell
bash scripts/preprocess/register.sh
```

Computes ground truth rotation parameters between pre- and post-orthodontic dental models.

> Adjust the data directory paths in all scripts above to match your setup.

## 2. Pretrain

Pretrain MeshMAE on remeshed tooth meshes:

```shell
bash scripts/main/pretrain.sh
```

Configure `config/pretrain.yaml`:
- `file`: path to a `.txt` file where each line is the full path to a remeshed single tooth mesh.

See https://github.com/liang3588/MeshMAE for more details on the MeshMAE architecture.

## 3. Train & Test

### 3.1 Train

```shell
bash scripts/main/train.sh
```

Configure `config/TADPM.yaml`:
- `dataroot`: directory containing remeshed data
- `paramroot`: directory containing rotation parameters (from step 1.4)
- `before_path` / `after_path`: point cloud directories before and after treatment

Specify the pretrained MeshMAE checkpoint via `--encoder_ckpts` in the script.

### 3.2 Test

```shell
bash scripts/main/test.sh
```

Specify both the main model checkpoint (`--ckpts`) and the encoder checkpoint (`--encoder_ckpts`).
