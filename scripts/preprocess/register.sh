ROOT=/data3/dtl/multimodal_teeth_dataset_1215

python -m data_preprocess.register \
    --single_before "$ROOT/remesh_before" \
    --single_after "$ROOT/remesh_after" \
    --outputroot "$ROOT/rotation_param" \
    --index_file "$ROOT/number_list.csv" \