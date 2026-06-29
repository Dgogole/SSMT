ROOT=/data/dtl/multimodal_teeth_dataset

DATAROOT=$ROOT/single_before
MANIFOLD=$ROOT/my_process/manifold_before
SIMPLIFY=$ROOT/my_process/simplify_before
REMESH=$ROOT/remesh_before

echo "Step 1/3: Manifold processing..."
python data_preprocess/manifold.py \
    --dataroot "$DATAROOT" \
    --manifold "$MANIFOLD" \
    --simplify "$SIMPLIFY"

echo "Step 2/3: Simplify processing..."
python data_preprocess/simplify.py \
    --dataroot "$DATAROOT" \
    --manifold "$MANIFOLD" \
    --simplify "$SIMPLIFY"

echo "Step 3/3: Data generation..."
python data_preprocess/datagen_maps.py \
    --simplify "$SIMPLIFY" \
    --output "$REMESH"