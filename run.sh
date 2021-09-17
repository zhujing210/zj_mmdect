#!/bin/bash

config_file=dgretinalnet_mb_fpn_1x_phone_money.py
work_dir=/mnt/datadisk0/jingzhudata/work_dirs/dgretinanet_mb_fpn_wallet_phonepy_original_aug
result_file=result.pkl
#output=/mnt/datadisk0/jingzhudata/badcase/badcase_wallet_money
#log_json = 20210802_205602.log.json
# Train
# CUDA_VISIBLE_DEVICES=6,7 PORT=29315 taskset -c 30-71 ./tools/dist_train.sh  configs/dg/${config_file} 2 --no-validate
#CUDA_VISIBLE_DEVICES=6,7 PORT=29325  taskset -c 30-71 ./tools/dist_train.sh  configs/dg/dg_mb_ssd_business_9_3.py 2 
# Test
#CUDA_VISIBLE_DEVICES=6,7 PORT=23282 ./tools/dist_test.sh configs/dg/${config_file} ${work_dir}/latest.pth 2 --out ${work_dir}/${result_file} --eval mAP
CUDA_VISIBLE_DEVICES=6,7 PORT=23282 ./tools/dist_test.sh configs/dg/${config_file} ${work_dir}/latest.pth 2  --eval mAP 

# loss 曲线
#python tools/analysis_tools/analyze_logs.py plot_curve ${work_dir}/20210805_101459.log.json --keys loss_cls loss_bbox loss --legend loss_cls loss_bbox loss --out ${work_dir}/losses.pdf

# FLOPs
# python tools/analysis_tools/get_flops.py configs/dg/${config_file} --shape 192 320

# Badcase
# 自己脚本分析 漏检误检
#python tools/analysis_tools/analyze_falseORundetected.py configs/dg/${config_file} ${work_dir}/${result_file} ${output} --show-score-thr 0.15 --re-hw 576 960 
#python tools/analysis_tools/analyze_falseORundetected.py configs/dg/${config_file} ${work_dir}/${result_file} ${output} --show-score-thr 0.15 --re-hw 192 320 --stastic-undetect

# mmdect 脚本，根据mAP 排序
# python tools/analysis_tools/analyze_results.py configs/dg/${config_file} results.pkl view_results/Dg --topk 500 --show-score-thr 0.15

# Visualize Datasets:
#python3 tools/misc/browse_dataset.py configs/dg/${config_file} --skip-type PhotoMetricDistortion Expand MinIoURandomCrop Expand2Canvas Normalize RandomFlip DefaultFormatBundle Collect --output-dir ${output} --not-show
