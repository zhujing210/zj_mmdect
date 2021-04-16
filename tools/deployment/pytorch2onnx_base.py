import argparse
from functools import partial

import mmcv
import numpy as np
import onnxruntime as rt
import torch
from mmcv.onnx import register_extra_symbolics
from mmcv.runner import load_checkpoint

# from mmcls.models import build_classifier
from mmdet.models import build_detector

torch.manual_seed(3)


def _demo_mm_inputs(input_shape, num_classes):
    """Create a superset of inputs needed to run test or train batches.

    Args:
        input_shape (tuple):
            input batch dimensions
        num_classes (int):
            number of semantic classes
    """
    (N, C, H, W) = input_shape
    rng = np.random.RandomState(0)
    imgs = rng.rand(*input_shape)
    gt_labels = rng.randint(
        low=0, high=num_classes - 1, size=(N, 1)).astype(np.uint8)
    mm_inputs = {
        'imgs': torch.FloatTensor(imgs).requires_grad_(True),
        'gt_labels': torch.LongTensor(gt_labels),
    }
    return mm_inputs


def pytorch2onnx(model,
                 input_shape,
                 opset_version=11,
                 show=False,
                 output_file='tmp.onnx',
                 verify=False):
    """Export Pytorch model to ONNX model and verify the outputs are same
    between Pytorch and ONNX.

    Args:
        model (nn.Module): Pytorch model we want to export.
        input_shape (tuple): Use this input shape to construct
            the corresponding dummy input and execute the model.
        opset_version (int): The onnx op version. Default: 11.
        show (bool): Whether print the computation graph. Default: False.
        output_file (string): The path to where we store the output ONNX model.
            Default: `tmp.onnx`.
        verify (bool): Whether compare the outputs between Pytorch and ONNX.
            Default: False.
    """
    try:
        from mmcv.onnx.symbolic import register_extra_symbolics
    except ModuleNotFoundError:
        raise NotImplementedError('please update mmcv to version>=v1.0.4')
    register_extra_symbolics(opset_version)

    model.cpu().eval()

    (N, C, H, W) = input_shape
    rng = np.random.RandomState(0)
    imgs = rng.rand(*input_shape)
    
    # cls
    # img_list = [img[None, :] for img in imgs] #tesnor format:1,3,300,300

    one_img = imgs[0, ::]
   
    show_img = one_img.transpose(1, 2, 0) #HWC

    one_img = torch.from_numpy(one_img).unsqueeze(0).float().requires_grad_(True)
    tensor_data = [one_img]

    one_meta = {
        'img_shape': (H, W, C),
        'ori_shape': (H, W, C),
        'pad_shape': (H, W, C),
        'filename': '<demo>.png',
        'scale_factor': 1.0,
        'flip': False,
        'show_img': show_img,
    }

    # replace original forward function
    origin_forward = model.forward
    model.forward = partial(model.forward, img_metas=[[one_meta]], return_loss=False)
    #cls
    # model.forward = partial(model.forward, return_loss=False)

    register_extra_symbolics(opset_version)
    with torch.no_grad():
        torch.onnx.export(
            model,
            tensor_data,
            output_file,
            input_names=['data'],
            # output_names=output_names,
            export_params=True,
            keep_initializers_as_inputs=True,
            do_constant_folding=True,
            verbose=show,
            opset_version=opset_version) #
        #cls
        # torch.onnx.export(
        #     model, (img_list, ),
        #     output_file,
        #     export_params=True,
        #     keep_initializers_as_inputs=True,
        #     verbose=show,
        #     opset_version=opset_version)
        print(f'Successfully exported ONNX model: {output_file}')
    # model.forward = origin_forward

    if verify:
        # check by onnx
        import onnx
        onnx_model = onnx.load(output_file)
        onnx.checker.check_model(onnx_model)

        # check the numerical value
        # get pytorch output
        #cls
        pytorch_result = model(tensor_data, return_loss=False)[0]

        # pytorch_result = model(tensor_data, img_metas=[[one_meta]], return_loss=False)[0]

        # get onnx output
        input_all = [node.name for node in onnx_model.graph.input]
        input_initializer = [
            node.name for node in onnx_model.graph.initializer
        ]
        net_feed_input = list(set(input_all) - set(input_initializer))
        assert (len(net_feed_input) == 1)
        sess = rt.InferenceSession(output_file)
        onnx_result = sess.run(
            None, {net_feed_input[0]: tensor_data[0].detach().numpy()})[0]
        print(pytorch_result, "\n debug,,,,, \n", onnx_result)
        if not np.allclose(pytorch_result, onnx_result):
            raise ValueError(
                'The outputs are different between Pytorch and ONNX')
        print('The outputs are same between Pytorch and ONNX')


def parse_args():
    parser = argparse.ArgumentParser(description='Convert MMCls to ONNX')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('--checkpoint', help='checkpoint file', default=None)
    parser.add_argument('--show', action='store_true', help='show onnx graph')
    parser.add_argument(
        '--verify', action='store_true', help='verify the onnx model')
    parser.add_argument('--output-file', type=str, default='tmp.onnx')
    parser.add_argument('--opset-version', type=int, default=11)
    parser.add_argument(
        '--shape',
        type=int,
        nargs='+',
        default=[224, 224],
        help='input image size')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    if len(args.shape) == 1:
        input_shape = (1, 3, args.shape[0], args.shape[0])
    elif len(args.shape) == 2:
        input_shape = (
            1,
            3,
        ) + tuple(args.shape)
    else:
        raise ValueError('invalid input shape')

    cfg = mmcv.Config.fromfile(args.config)
    cfg.model.pretrained = None

    # build the model and load checkpoint
    model = build_detector(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))

    if args.checkpoint:
        load_checkpoint(model, args.checkpoint, map_location='cpu')

    # conver model to onnx file
    pytorch2onnx(
        model,
        input_shape,
        opset_version=args.opset_version,
        show=args.show,
        output_file=args.output_file,
        verify=args.verify)
