"""作用：实现 LLaVA 在对应下游任务上的评测、答案读取、指标计算或结果转换逻辑。"""

import os
import argparse
import json
import numpy as np
from PIL import Image
import pandas as pd


def get_args():
    """作用：读取、筛选或组装指定对象并返回给调用方。"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--test-file', type=str, default='playground/Instructions_slim/Grounding/test.json')
    parser.add_argument('--result-file', type=str,
                        default='results/CoIN_normaltrain_testslim/Grounding/OCRVQA/merge.jsonl')
    parser.add_argument('--output-dir', type=str)
    return parser.parse_args()


def expand2square(pil_img, background_color):
    """作用：执行 expand2square 函数对应的工具逻辑，供当前脚本或其他模块复用。"""
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


def change_bbox(bbox, im_w, im_h):
    """作用：执行 change_bbox 函数对应的工具逻辑，供当前脚本或其他模块复用。"""
    x, y, w, h = bbox
    x1, y1, x2, y2 = x, y, x + w, y + h
    max_wh = max(im_w, im_h)
    if im_w == im_h:
        return [x / max_wh, y / max_wh, x2 / max_wh, y2 / max_wh]
    elif im_w > im_h:
        y1 = y1 + (im_w - im_h) / 2
        y2 = y2 + (im_w - im_h) / 2
        return [x1 / max_wh, y1 / max_wh, x2 / max_wh, y2 / max_wh]
    else:
        x1 = x1 + (im_h - im_w) / 2
        x2 = x2 + (im_h - im_w) / 2
        return [x1 / max_wh, y1 / max_wh, x2 / max_wh, y2 / max_wh]


def calculate_iou(bbox1, bbox2):
    """作用：根据输入数值执行公式计算并返回结果。"""
    x1, y1, x2, y2 = bbox1
    x21, y21, x22, y22 = bbox2
    intersection_area = max(0, min(x2, x22) - max(x1, x21)) * max(0, min(y2, y22) - max(y1, y21))
    union_area = (x2 - x1) * (y2 - y1) + (x22 - x21) * (y22 - y21) - intersection_area
    iou = intersection_area / union_area if union_area > 0 else 0
    return iou


def eval_single(test_file, result_file):
    """作用：执行指定任务的评测流程并输出指标或结果文件。"""
    annotations = json.load(open(test_file))
    annotations = {grounding_test['question_id']: grounding_test for grounding_test in annotations}
    results = [json.loads(line) for line in open(result_file)]
    # breakpoint()
    pred_list = []
    total = len(results)
    right = 0
    # 收集所有IoU值
    iou_values = []
    valid_samples = 0
    total_samples = len(results)
    for result in results:
        grounding_gt = annotations[result['question_id']]
        bbox_string = grounding_gt['answer_bbox']
        bbox_string = bbox_string.replace('[', '').replace(']', '')
        bbox_groundtruth = [float(x) for x in bbox_string.split(',')]
        size = grounding_gt['size']

        pred_bbox = result['text']
        try:
            pred_bbox = pred_bbox.replace('[', '').replace(']', '')
            bbox_pred = [float(x) for x in pred_bbox[1:-1].split(',')]
            if len(bbox_pred) != 4:
                continue
        except:
            continue

        max_wh = max(size)
        bbox_pred = [x * max_wh for x in bbox_pred]
        bbox_groundtruth = [x * max_wh for x in bbox_groundtruth]

        iou = calculate_iou(bbox_pred, bbox_groundtruth)
        right += iou > 0.5

        iou_values.append(iou)
        valid_samples += 1

    # 转换为numpy数组便于分析
    iou_array = np.array(iou_values)
    
    # 计算统计指标
    stats = {
        'total_samples': total_samples,
        'valid_samples': valid_samples,
        'mean_iou': np.mean(iou_array),
        'median_iou': np.median(iou_array),
        'std_iou': np.std(iou_array),
        'min_iou': np.min(iou_array),
        'max_iou': np.max(iou_array),
        'accuracy_0.3': np.mean(iou_array > 0.3),
        'accuracy_0.5': np.mean(iou_array > 0.5),
        'accuracy_0.7': np.mean(iou_array > 0.7),
        'accuracy_0.9': np.mean(iou_array > 0.9),
    }
    save_detailed_results(iou_values, stats, args.output_dir)

    print('Samples: {}\nAccuracy: {:.2f}%\n'.format(total, 100. * right / total))
    
    if args.output_dir is not None:
        output_file = os.path.join(args.output_dir, 'Result.text')
        with open(output_file, 'w') as f:
            f.write('Samples: {}\nAccuracy: {:.2f}%\n'.format(total, 100. * right / total))
            f.write("IoU统计分析报告\n")
            f.write("=" * 50 + "\n")
            for key, value in stats.items():
                if isinstance(value, float):
                    f.write(f"{key}: {value:.4f}\n")
                else:
                    f.write(f"{key}: {value}\n")

def save_detailed_results(iou_values, stats, output_dir):
    """保存详细结果到文件"""
    
    # 保存每个样本的IoU值
    iou_df = pd.DataFrame({
        'sample_id': range(len(iou_values)),
        'iou': iou_values
    })
    iou_df.to_csv(os.path.join(output_dir, 'detailed_iou_values.csv'), index=False)

if __name__ == "__main__":
    args = get_args()

    if args.result_file is not None:
        eval_single(args.test_file, args.result_file)
