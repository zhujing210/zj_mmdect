### mmcls

# python tools/train.py configs/lenet/lenet5_mnist.py --gpus 1

# python tools/train.py configs/dg/resnet_cds.py --gpus 1 --no-validate 

# CUDA_VISIBLE_DEVICES=3 python tools/train.py configs/dg/squeezenet_bn_lmk_pose_occ.py --gpus 1 --no-validate


#CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29500
# CUDA_VISIBLE_DEVICES=2,3 PORT=29501 ./tools/dist_train.sh configs/dg/squeezenet_bn_lmk_pose_occ.py 2 --no-validate

# CUDA_VISIBLE_DEVICES=0 python tools/test.py configs/dg/squeezenet_bn_lmk_pose_occ.py work_dirs/squeezenet_bn_lmk_pose_occ/epoch_140.pth --metrics accuracy



### mmdet
# CUDA_VISIBLE_DEVICES=1,2,3 PORT=29501 ./tools/dist_train.sh configs/dg/ssd300_voc.py 3 --no-validate
# CUDA_VISIBLE_DEVICES=1,0,2,3 PORT=29501 ./tools/dist_train.sh configs/dg/ssd300_fire.py 4 --no-validate

# CUDA_VISIBLE_DEVICES=3 python tools/train.py configs/dg/ssd300_fire.py --gpus 1 --no-validate 


# is ok
# CUDA_VISIBLE_DEVICES=1,0,2,3 PORT=29501 ./tools/dist_train.sh configs/dg/ssd300_coco.py 4 --no-validate


#pytorch2onnx
# python tools/deployment/pytorch2onnx.py \
#     configs/dg/ssd300_voc.py \
#     work_dirs/ssd300_voc/epoch_5.pth \
#     --output-file work_dirs/ssd300_voc/tmp.onnx \
#     --input-img demo/demo.jpg \
#     --test-img tests/data/color.jpg \
#     --shape 300 300 \
#     --mean 123.675 116.28 103.53 \
#     --std 1 1 1 \
#     --show \
#     --verify \
#     --simplify \
#     --opset-version 12


python tools/deployment/pytorch2onnx_base.py \
    configs/dg/ssd300_voc.py \
    --checkpoint work_dirs/ssd300_voc/epoch_5.pth \
    --output-file work_dirs/ssd300_voc/tmp.onnx \
    --shape 300 300 \
    --verify \
    --show \
    --opset-version 11