import argparse
import os.path as osp   

import mmcv
import numpy as np
from mmcv import Config, DictAction

from mmdet.core.evaluation.undet_false_det import upfp_singleclass, iou_acrossclass
from mmdet.core.visualization import imshow_gt_det_bboxes
from mmdet.datasets import build_dataset, get_loading_pipeline



class UndectFalseDetect(object):
    """Display and save evaluation results.

    Args:
        show (bool): Whether to show the image. Default: True
        wait_time (float): Value of waitKey param. Default: 0.
        score_thr (float): Minimum score of bboxes to be shown.
           Default: 0
    """

    def __init__(self, show=False, wait_time=0, score_thr=0.15, re_hw=[192, 320], stastic_undetect=True):
        self.show = show
        self.wait_time = wait_time
        self.score_thr = score_thr
        self.re_hw = re_hw
        self.stastic_undetect = stastic_undetect
    
    def get_cls_results(self, det_results, annotations, class_id):
        """Get det results and gt information of a certain class.

        Args:
            det_results (list[list]): Same as `eval_map()`.
            annotations (list[dict]): Same as `eval_map()`.
            class_id (int): ID of a specific class.

        Returns:
            tuple[list[np.ndarray]]: detected bboxes, gt bboxes, ignored gt bboxes
        """
        cls_dets = [img_res[class_id] for img_res in det_results]
        cls_gts = []
        cls_gts_ignore = []
        for ann in annotations:
            gt_inds = ann['labels'] == class_id
            cls_gts.append(ann['bboxes'][gt_inds, :])

            if ann.get('labels_ignore', None) is not None:
                ignore_inds = ann['labels_ignore'] == class_id
                cls_gts_ignore.append(ann['bboxes_ignore'][ignore_inds, :])
            else:
                cls_gts_ignore.append(np.empty((0, 4), dtype=np.float32))

        return cls_dets, cls_gts, cls_gts_ignore
    
    def udfd(self, det_results, annotations, img_shape, file_name):
        """"
            det_results: img(class(bbox))
        """
        assert len(det_results) == len(annotations)
        num_classes = len(det_results[0])  # positive class num
        for i in range(num_classes):
            cls_dets, cls_gts, cls_gts_ignore = self.get_cls_results(det_results, annotations, i)
            if upfp_singleclass(cls_dets[0], cls_gts[0], img_shape, self.re_hw,
                                 self.stastic_undetect, file_name, cls_gts_ignore[0]):
                return True
            num_gts = cls_gts[0].shape[0]
            if num_gts > 0:
                ious = np.array([])
                for j in range(num_classes):
                    cls_dets2, _, _ = self.get_cls_results(det_results, annotations, j)
                    iou = iou_acrossclass(cls_dets2[0], cls_gts[0], cls_gts_ignore[0], high_score=0.15)
                    ious = iou if ious.size==0 else np.vstack((ious, iou))
                if (ious.max(axis=0) > iou).any():
                    return True
        return False


    def _save_image_gts_results(self, data_info, classes, results, out_dir=None):
        mmcv.mkdir_or_exist(out_dir)
        # calc save file path
        filename = data_info['filename']

        if data_info['img_prefix'] is not None:
                filename = osp.join(data_info['img_prefix'], filename)
        else:
            filename = data_info['filename']
        filename = osp.basename(filename)
        out_file = osp.join(out_dir, filename)
        # print("*"*10)
        # print(out_file)
        # print("*"*10)
        # exit()       
        imshow_gt_det_bboxes(
            data_info['img'],
            data_info,
            results,
            classes,
            show=self.show,
            score_thr=self.score_thr,
            wait_time=self.wait_time,
            out_file=out_file)

    def evaluate_and_show(self,
                          dataset,
                          results,
                          show_dir):
        """Evaluate and show results.

        Args:
            dataset (Dataset): A PyTorch dataset.
            results (list): Det results from test results pkl file
            show_dir (str, optional): The filename to write the image.
                Default: 'work_dir'
        """

        prog_bar = mmcv.ProgressBar(len(results))
        for i, (result, ) in enumerate(zip(results)):
            # self.dataset[i] should not call directly
            # because there is a risk of mismatch
            data_info = dataset.prepare_train_img(i)
            img_shape = data_info['img'].shape
            file_name = data_info['img_info']['filename']
            
            if isinstance(result, tuple):
                bbox_det_result = [result[0]]
            else:
                bbox_det_result = [result]
           
            udfd = self.udfd(bbox_det_result , [data_info['ann_info']], img_shape, file_name)
            if udfd:
                if not self.stastic_undetect:
                    self._save_image_gts_results(data_info, dataset.CLASSES, results[i], show_dir)
            prog_bar.update()


def parse_args():
    parser = argparse.ArgumentParser(
        description='zj eval image prediction result for each')
    parser.add_argument('config', help='test config file path')
    parser.add_argument(
        'prediction_path', help='prediction path where test pkl result')
    parser.add_argument(
        'show_dir', help='directory where painted images will be saved')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--wait-time',
        type=float,
        default=0,
        help='the interval of show (s), 0 is block')
    parser.add_argument(
        '--re-hw',
        type=int,
        nargs='+',
        default=[192, 320],
        help='convas size')
    parser.add_argument(
        '--show-score-thr',
        type=float,
        default=0.15,
        help='score threshold (default: 0.15)')
    parser.add_argument(
        '--stastic-undetect',
        action='store_true', 
        help='just calculate undect , do not plot')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    mmcv.check_file_exist(args.prediction_path)

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    cfg.data.test.test_mode = True
    # import modules from string list.
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])

    cfg.data.test.pop('samples_per_gpu', 0)
    cfg.data.test.pipeline = get_loading_pipeline(cfg.data.train.pipeline)
    dataset = build_dataset(cfg.data.test)
    outputs = mmcv.load(args.prediction_path)

    result_visualizer = UndectFalseDetect(args.show, args.wait_time,
                                         args.show_score_thr, args.re_hw, args.stastic_undetect)
    result_visualizer.evaluate_and_show(
        dataset, outputs,  show_dir=args.show_dir)


if __name__ == '__main__':
    main()
