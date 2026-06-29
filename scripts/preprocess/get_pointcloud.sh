python data_preprocess/get_pointcloud.py \
    --before_dataroot '/data3/dtl/multimodal_teeth_dataset/single_before' \
    --after_dataroot '/data3/dtl/multimodal_teeth_dataset/single_after' \
    --before_output '/data3/dtl/multimodal_teeth_dataset/pcd_before' \
    --after_output '/data3/dtl/multimodal_teeth_dataset/pcd_after' \
    --sample_num 512

# before_dataroot: data root for single tooth before treatment
# after_dataroot: data root for single tooth after treatment
# before_output: pointcloud output for single tooth before treatment
# after_output: pointcloud output for single tooth after treatment