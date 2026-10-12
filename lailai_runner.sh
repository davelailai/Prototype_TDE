# for Magin_dir in work_dirs_CounterFactual_TDE/FFA/Res50_swav_448_lr5e-3_AUC/CounterFactual_TDE_New/Query_with_project_gate_new_ablation/*
# do
#     [ -d "$Magin_dir" ] || continue
#     echo "Processing Margin Directory: $Magin_dir"
#     for METHOD_DIR in $Magin_dir/*
#     do
#         # 跳过非文件夹
#         [ -d "$METHOD_DIR" ] || continue
#         # # 跳过指定文件夹，比如 Asy    
#         # if [ "$(basename "$METHOD_DIR")" = "Asy" ]; then
#         #     continue
#         # fi
#         # METHOD_DIR=work_dirs_CounterFactual_TDE/FFA/Res50_swav_448_lr5e-3_AUC/BaseLine/Q2L
#         for fold in 1 2 3 4 5
#         do
#             LOOP_DIR="$METHOD_DIR/loop_${fold}"
#             for CONFIG in $LOOP_DIR/*.py
#             do
#                 WORK_DIR="$LOOP_DIR"
#                 CHECKPOINT=$(ls $WORK_DIR/best_multi-class_mAUC_multiclass_epoch*.pth | head -n 1)
#                 GPUS=1  # set as needed         

#                 if [ -f "$CHECKPOINT" ]; then
#                     bash tools/dist_test.sh $CONFIG $CHECKPOINT $GPUS --work-dir=$WORK_DIR --out $WORK_DIR/results_$(basename $CONFIG .py).pkl --out-item pred 
#                 else
#                     echo "Checkpoint not found for fold $fold in $WORK_DIR"
#                 fi
#             done
#         done
#     done
# done
# for METHOD_DIR in work_dirs_CounterFactual_TDE/FFA/Res50_swav_448_lr5e-3_AUC/CounterFactual_TDE_New/Query_with_project_gate_new_ablation/*
# do
#     # 跳过非文件夹
#     [ -d "$METHOD_DIR" ] || continue
#     # # 跳过指定文件夹，比如 Asy    
#     # if [ "$(basename "$METHOD_DIR")" = "Asy" ]; then
#     #     continue
#     # fi
# # METHOD_DIR=work_dirs_CounterFactual_TDE/ODIR/Res50_swav_448_lr5e-3_AUC/BaseLine/Q2L
#     for fold in 1 2 3 4 5
#     do
#         LOOP_DIR="$METHOD_DIR/margin0.7/loop_${fold}"
#         for CONFIG in $LOOP_DIR/*.py
#         do
#             WORK_DIR="$LOOP_DIR"
#             CHECKPOINT=$(ls $WORK_DIR/best_multi-class_mAUC_multiclass_epoch*.pth | head -n 1)
#             GPUS=1  # set as needed

#             if [ -f "$CHECKPOINT" ]; then
#                 bash tools/dist_test.sh $CONFIG $CHECKPOINT $GPUS --work-dir=$WORK_DIR --out $WORK_DIR/results_$(basename $CONFIG .py).pkl --out-item pred 
#             else
#                 echo "Checkpoint not found for fold $fold in $WORK_DIR"
#             fi
#         done
#     done
# done
# BASE_DIR='work_dirs_BankCausal/Endo/BankCausal_rebuttal/BankCausal_rebuttal/Co_Ca_ratio_new'
# for run_dir in "$BASE_DIR"/run_*; do
#     [ -d "$run_dir" ] || continue
#     for method_dir in "$run_dir"/*; do
#         if [ -d "$method_dir" ]; then
#             config=$(ls "$method_dir"/*.py 2>/dev/null | head -n 1)
#             ckpt=$(ls "$method_dir"/best*.pth 2>/dev/null | head -n 1)
#             if [ -f "$config" ] && [ -f "$ckpt" ]; then
#                 echo "Testing $ckpt with $config in $method_dir"
#                 python tools/Test.py "$config" "$ckpt" \
#                     --work-dir "$method_dir/eval" \
#                     --cfg-options custom_hooks.1.save_path="$method_dir/eval/Clustering.png" --tta
#             else
#                 echo "No config or checkpoint found in $method_dir"
#             fi
#         fi
#     done
# done

# BASE_DIR='work_dirs_BankCausal/Endo/BankCausal_rebuttal/BankCausal_rebuttal/backbone'
# for backbone_dir in "$BASE_DIR"/*; do
#     [ -d "$backbone_dir" ] || continue
#     echo "Processing backbone: $backbone_dir"
#     for method_dir in "$backbone_dir"/*; do
#         if [ -d "$method_dir" ]; then
#             config=$(ls "$method_dir"/*.py 2>/dev/null | head -n 1)
#             ckpt=$(ls "$method_dir"/best*.pth 2>/dev/null | head -n 1)
#             if [ -f "$config" ] && [ -f "$ckpt" ]; then
#                 echo "Testing $ckpt with $config in $method_dir"
#                 python tools/Test.py "$config" "$ckpt" \
#                     --work-dir "$method_dir/eval" \
#                     --tta
#                     # --cfg-options custom_hooks.1.save_path="$method_dir/eval/Clustering.png" --tta
#             else
#                 echo "No config or checkpoint found in $method_dir"
#             fi
#         fi
#     done
# done

# BASE_DIR='work_dirs_BankCausal/ChestX_Pert_new/SGD_5e-3/BankCausal_rebuttal/release_Skin_2'
# # for backbone_dir in "$BASE_DIR"/*; do
# #     [ -d "$backbone_dir" ] || continue
# #     echo "Processing backbone: $backbone_dir"
# for method_dir in "$BASE_DIR"/*; do
#     if [ -d "$method_dir" ]; then
#         config=$(ls "$method_dir"/*.py 2>/dev/null | head -n 1)
#         ckpt=$(ls "$method_dir"/best*.pth 2>/dev/null | head -n 1)
#         if [ -f "$config" ] && [ -f "$ckpt" ]; then
#             echo "Testing $ckpt with $config in $method_dir"
#             python tools/Test.py "$config" "$ckpt" \
#                 --work-dir "$method_dir/eval" \
#                 --cfg-options custom_hooks.1.save_path="$method_dir/eval/Clustering.png"
#         else
#             echo "No config or checkpoint found in $method_dir"
#         fi
#     fi
# done

# BASE_DIR='work_dirs_BankCausal/Endo/BankCausal_rebuttal/BankCausal_rebuttal/release_Skin_new'

# #!/usr/bin/env bash

BASE_DIR='work_dirs/work_dirs_Prototype_TDE_0602/FFA/Lambda_Ablation_margin0.5_new'

for method_dir in "$BASE_DIR"/*; do
    [ -d "$method_dir" ] || continue

    echo "========== Processing method: $method_dir =========="

    for fold_idx in 0 1 2 3 4; do
        fold_dir="$method_dir/fold${fold_idx}"

        if [ ! -d "$fold_dir" ]; then
            echo "Skip missing fold directory: $fold_dir"
            continue
        fi

        config=$(find "$fold_dir" -maxdepth 1 -type f -name '*.py' | head -n 1)
        ckpt=$(find "$fold_dir" -maxdepth 1 -type f -name 'best*.pth' | head -n 1)

        if [ -f "$config" ] && [ -f "$ckpt" ]; then
            eval_dir="$fold_dir/eval"
            mkdir -p "$eval_dir"

            echo "Testing fold_${fold_idx}: $ckpt with $config"
            OUT_FILE="$eval_dir/pred_results_fold${fold_idx}.pkl"
            python tools/Test.py "$config" "$ckpt" \
                --out "$OUT_FILE" \
                --work-dir "$eval_dir" \
                --out-item pred
                # --cfg-options custom_hooks.1.save_path="$eval_dir/Clustering.png"
        else
            echo "No config or checkpoint found in $fold_dir"
        fi
    done

    echo "========== Finished method: $method_dir =========="
done

# DATASETS=("BankCausal_rebuttal")

# # 在这里填写希望运行的子目录（相对 BASE_DIR）。留空表示遍历全部。
# TARGET_SUBDIRS=(
#     'release_Skin_new'
#     # 'proto2_globalFalse' 'proto2_globalTrue' 'Q2L' 'Q2LCausal'

# )
# # 只跑指定方法目录（如 Q2L）。留空则遍历 TARGET_SUBDIRS 下所有方法。
# TARGET_METHOD=()
# process_dir() {
#     local sub_dir="$1"
#     [ -d "$sub_dir" ] || return

#     sub_dir_name=$(basename "$sub_dir")
#     echo "========== Processing directory: $sub_dir_name =========="
    
#     for METHOD_DIR in "$sub_dir"/*; do
#         [ -d "$METHOD_DIR" ] || continue
#         METHOD=$(basename "$METHOD_DIR")
#         # 如果设置了 TARGET_METHOD，跳过不匹配的方法目录
#         [ -n "$TARGET_METHOD" ] && [ "$METHOD" != "$TARGET_METHOD" ] && continue
#         echo "  Method: $METHOD"
        
#         for fold in 1 2 3 4 5; do
#             WORK_DIR="$METHOD_DIR/fold$((fold-1))"
#             [ -d "$WORK_DIR" ] || continue
            
#             echo "    Fold $fold"
            
#             Check whether eval json already exists
#             json_count=$(find "$WORK_DIR/eval" -type f -name "*.json" 2>/dev/null | wc -l)
#             if [ "$json_count" -gt 0 ]; then
#                 echo "      Found $json_count eval/*.json, skip test."
#                 continue
#             fi

#             CONFIG=$(ls "$WORK_DIR"/*.py 2>/dev/null | head -n 1)
#             CHECKPOINT=$(ls "$WORK_DIR"/best_multi-class*.pth 2>/dev/null | head -n 1)
            
#             # 如果没有找到best checkpoint，则尝试使用last epoch checkpoint
#             if [ -z "$CHECKPOINT" ]; then
#                 CHECKPOINT=$(ls "$WORK_DIR"/epoch_*.pth 2>/dev/null | tail -n 1)
#                 [ -n "$CHECKPOINT" ] && echo "      Using last epoch checkpoint"
#             fi
            
#             if [ -f "$CHECKPOINT" ] && [ -f "$CONFIG" ]; then
#                 mkdir -p "$WORK_DIR/eval"
#                 echo "      Testing with checkpoint: $(basename "$CHECKPOINT")"
#                 OUT_FILE="$WORK_DIR/eval/pred_results_fold${fold}.pkl"
#                 python -u tools/Test.py "$CONFIG" "$CHECKPOINT" \
#                     --out "$OUT_FILE" \
#                     --work-dir "$WORK_DIR/eval" \
#                     --out-item pred \
#                     --cfg-options custom_hooks.1.save_path="$WORK_DIR/eval/Clustering.png"
#                 echo "      Saved predictions to $OUT_FILE"
#             else
#                 echo "      Config or checkpoint not found"
#                 [ -z "$CONFIG" ] && echo "        Missing: *.py config file"
#                 [ -z "$CHECKPOINT" ] && echo "        Missing: best_multi-label_mAUC_epoch*.pth"
#             fi
#         done
#     done
#     echo "========== Finished directory: $sub_dir_name =========="
# }

# for dataset in "${DATASETS[@]}"; do
#     BASE_DIR="work_dirs_BankCausal/Endo/BankCausal_rebuttal/${dataset}"
#     [ -d "$BASE_DIR" ] || { echo "Skip missing BASE_DIR: $BASE_DIR"; continue; }

#     if [ ${#TARGET_SUBDIRS[@]} -gt 0 ]; then
#         for rel in "${TARGET_SUBDIRS[@]}"; do
#             process_dir "$BASE_DIR/$rel"
#         done
#     else
#         for sub_dir in "$BASE_DIR"/*; do
#             [ -d "$sub_dir" ] || continue
#             process_dir "$sub_dir"
#         done
#     fi
# done


# BASE_DIR="work_dirs/MedMNIST_runs/2D/Q2L_counterFactual-TDE"
# for MARGIN in 0.3 0.5 0.7 0.9 1.1
# do
#     for DATASET_DIR in $BASE_DIR/Margin$MARGIN/*
#     do
#         [ -d "$DATASET_DIR" ] || continue
#         # 只处理 retinamnist
#         # if [ "$(basename "$DATASET_DIR")" != "tissuemnist" ]; then
#         #     continue
#         # fi
#         for fold in 1 2 3 4 5
#         do
#             LOOP_DIR="$DATASET_DIR/run_${fold}"
#             for CONFIG in $LOOP_DIR/*.py
#             do
#                 WORK_DIR="$LOOP_DIR"
#                 CHECKPOINT=$(ls $WORK_DIR/best_multi-class_mAUC_multiclass_epoch*.pth | head -n 1)
#                 GPUS=1  # set as needed
#                 echo "WORK_DIR: $WORK_DIR"
#                 if [ -f "$CHECKPOINT" ]; then
#                     bash tools/dist_test.sh $CONFIG $CHECKPOINT $GPUS --work-dir=$WORK_DIR --out $WORK_DIR/results_$(basename $CONFIG .py).pkl --out-item pred 
#                 else
#                     echo "Checkpoint not found for fold $fold margin $MARGIN in $WORK_DIR"
#                 fi
#             done
#         done
#     done
# done

# echo "========================================================"
# echo "SLURM job submitted on: $(date)"
# date_start=$(date +%s)
# hostname
# echo "CUDA_VISIBLE_DEVICES : $CUDA_VISIBLE_DEVICES"
# echo "GPU_DEVICE_ORDINAL   : $GPU_DEVICE_ORDINAL"
# echo "Running robustness validation loop..."

# # Set fixed parameters
# CONFIG_FILE='configs/FFA/FFA_Q2L_Constrain.py'
# BASE_DIR='/media/jbridge/Expansion/code/Multi-disease-diagnosis'
# LOOP_PREFIX='work_dirs_MIL/FFA/Res50_swav_448_lr5e-3_AUC/Q2L_constrain/Q2L_with_independence/loop'
# SPARSE_WEIGHT=0
# ATTENTION_WEIGHT=0
# SMOOTH_WEIGHT=0
# prototype_init='query_mlp'
# # 'query_mlp' or 'independent'
# intervention_margin=0.05
# inervention_mode='gate'
# sparse_attention_loss_weight=0,
# attention_independence_loss_weight=0,
# spatial_smooth_weight=0,
# query_independence_loss_weight=0.1,
# binary_constraint_loss_weight=0.1,
# # Run loop
# for i in {1..5}
# do
#     RUN_DIR="${BASE_DIR}/${LOOP_PREFIX}_${i}"
         
#     echo "---- Run $i ----"
#     echo "Using:"
#     echo "  sparse_attention_loss_weight = 0"
#     echo "  attention_independence_loss_weight = 0"
#     echo "  spatial_smooth_weight = 0"
#     echo "  query_independence_loss_weight = 0.1"
#     echo "  binary_constraint_loss_weight = 0.1"
#     echo "  prototype_init = ${prototype_init}"
#     echo "  intervention_margin = ${intervention_margin}"
#     echo "  work_dir = ${RUN_DIR}"

#     python ${BASE_DIR}/tools/Train.py $CONFIG_FILE \
#         --work-dir $RUN_DIR \
#         --cfg-options \
#         model.head.task_heads.classification.sparse_attention_loss_weight=0 \
#         model.head.task_heads.classification.attention_independence_loss_weight=0 \
#         model.head.task_heads.classification.spatial_smooth_weight=0 \
#         model.head.task_heads.classification.query_independence_loss_weight=0.1 \
#         model.head.task_heads.classification.binary_constraint_loss_weight=0.1
#     echo "Run $i completed."
# done