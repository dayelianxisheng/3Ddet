"""
PointNet 预测脚本
支持批量推理和单张点云预测
"""

import os
import sys
import torch
import numpy as np
import importlib
from pathlib import Path
import argparse
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, os.path.join(ROOT_DIR, 'model'))

from dataset.ModelNetDataLoader import ModelNetDataLoader


def pc_normalize(pc):
    """归一化点云到单位球"""
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc


def load_point_cloud(file_path, num_point=1024, use_normals=False):
    """
    加载单个点云文件
    支持格式: .txt (x,y,z[,nx,ny,nz])
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f'Point cloud file not found: {file_path}')

    points = np.loadtxt(file_path, delimiter=',').astype(np.float32)

    # 归一化
    points[:, 0:3] = pc_normalize(points[:, 0:3])

    # 采样或补齐
    if points.shape[0] > num_point:
        indices = np.random.choice(points.shape[0], num_point, replace=False)
        points = points[indices]
    elif points.shape[0] < num_point:
        repeat_times = (num_point // points.shape[0]) + 1
        points = np.tile(points, (repeat_times, 1))[:num_point]

    # 是否使用法向量
    if not use_normals:
        points = points[:, 0:3]
    else:
        if points.shape[1] < 6:
            raise ValueError(f'Point cloud has {points.shape[1]} channels, but use_normals=True requires at least 6')
        points = points[:, :6]

    return points

def parse_args():
    """
      # 批量预测（默认）
      python tools/pointnet/predict_cls.py

      # 指定模型和数据划分
      python tools/pointnet/predict_cls.py --split test --batch_size 32

      # 单张预测
      python tools/pointnet/predict_cls.py --point_cloud path/to/sample.txt

      # 完整参数
      python tools/pointnet/predict_cls.py \
          --point_cloud path/to/sample.txt \
          --num_category 40 \
          --num_point 1024 \
          --log_dir 2026-04-01_19-33
        :return:
    """
    parser = argparse.ArgumentParser('PointNet Prediction')
    parser.add_argument('--use_cpu', action='store_true', default=False, help='use cpu mode')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--batch_size', type=int, default=24, help='batch size in batch mode')
    parser.add_argument('--num_category', default=40, type=int, choices=[10, 40], help='ModelNet10/40')
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number')
    parser.add_argument('--log_dir', type=str, default=None, help='Experiment root (default: latest)')
    parser.add_argument('--use_normals', action='store_true', default=False, help='use normals')

    # 单张预测模式
    parser.add_argument('--point_cloud', type=str, default=None, help='Path to .txt point cloud file (single mode)')
    parser.add_argument('--num_votes', type=int, default=1, help='Number of votes for single mode')

    # 批量预测模式
    parser.add_argument('--split', type=str, default='test', choices=['train', 'test'], help='dataset split (batch mode)')
    return parser.parse_args()


def get_latest_log_dir():
    exp_base = Path(ROOT_DIR) / 'tools' / 'pointnet' / 'log' / 'classification'
    if not exp_base.exists():
        raise FileNotFoundError(f'log directory not found: {exp_base}')
    candidates = sorted([d for d in exp_base.iterdir() if d.is_dir()], key=lambda d: d.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f'No log directory found in {exp_base}')
    return candidates[-1]


def load_classifier(log_dir, num_category, use_normals, device):
    model = importlib.import_module('pointnet_cls')
    classifier = model.get_model(num_category, normal_channel=use_normals)
    classifier = classifier.to(device)
    classifier.eval()
    checkpoint = torch.load(str(log_dir / 'checkpoints' / 'best_model.pth'), map_location=device, weights_only=False)
    classifier.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    return classifier


def predict_single(args, device, classifier, class_names):
    """单张点云预测"""
    # 加载点云
    if args.point_cloud is None:
        raise ValueError('--point_cloud is required in single prediction mode')
    print(f'Loading point cloud: {args.point_cloud}')
    points = load_point_cloud(args.point_cloud, num_point=args.num_point, use_normals=args.use_normals)
    print(f'Point cloud shape: {points.shape}')  # (N, C)

    # 转为 Tensor [1, C, N]  (DataLoader 输出是 [B, N, C]，需 transpose 成 [B, C, N])
    points_tensor = torch.from_numpy(points).unsqueeze(0).transpose(2, 1).to(device)

    # 推理
    with torch.no_grad():
        pred, _ = classifier(points_tensor)
        probs = torch.exp(pred.data)
        pred_idx = pred.data.max(1)[1].item()
        confidence = probs[0, pred_idx].item()

    # 输出结果
    pred_name = class_names[pred_idx]
    print(f'\n{"=" * 40}')
    print(f'Prediction:  {pred_name}')
    print(f'Confidence:  {confidence:.4f}')
    print(f'{"=" * 40}')

    # Top-5
    top5_probs, top5_indices = torch.topk(probs[0], min(5, args.num_category))
    print('\nTop-5 predictions:')
    for i, (idx, prob) in enumerate(zip(top5_indices, top5_probs)):
        print(f'  {i + 1}. {class_names[idx.item()]:20s}  {prob.item():.4f}')

    return pred_name, confidence


def predict_batch(args, device, classifier, class_names):
    """批量预测"""
    # 加载数据集
    args_obj = argparse.Namespace(
        num_point=args.num_point,
        use_uniform_sample=False,
        use_normals=args.use_normals,
        num_category=args.num_category,
        process_data=False
    )
    data_path = r'D:\resource\data\3d\modelnet40_normal_resampled'
    dataset = ModelNetDataLoader(root=data_path, args=args_obj, split=args.split, process_data=False)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # 推理
    print(f'\nStart inference on {len(dataset)} samples...')
    results = []
    with torch.no_grad():
        for points, labels in tqdm(dataloader, total=len(dataloader)):
            points, labels = points.to(device), labels.to(device)
            points = points.transpose(2, 1)
            pred, _ = classifier(points)
            pred_choice = pred.data.max(1)[1]
            probs = torch.exp(pred.data.max(1)[0])

            for i in range(len(pred_choice)):
                pred_idx = pred_choice[i].item()
                prob = probs[i].item()
                true_idx = labels[i].item()
                pred_name = class_names[pred_idx]
                true_name = class_names[true_idx]
                results.append({
                    'pred_idx': pred_idx,
                    'pred_name': pred_name,
                    'true_idx': true_idx,
                    'true_name': true_name,
                    'confidence': prob,
                    'correct': pred_idx == true_idx
                })

    # 统计
    total = len(results)
    correct = sum(1 for r in results if r['correct'])
    accuracy = correct / total * 100
    print(f'\n{"=" * 50}')
    print(f'Total: {total}, Correct: {correct}, Accuracy: {accuracy:.2f}%')
    print(f'{"=" * 50}')

    # 错误样本
    errors = [r for r in results if not r['correct']]
    if errors:
        print(f'\nMisclassified samples ({len(errors)}):')
        for r in errors[:10]:
            print(f'  True: {r["true_name"]:20s}  Pred: {r["pred_name"]:20s}  Conf: {r["confidence"]:.4f}')
        if len(errors) > 10:
            print(f'  ... and {len(errors) - 10} more')

    return results


def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() and not args.use_cpu else 'cpu')

    # 加载类别名称
    data_path = r'D:\resource\data\3d\modelnet40_normal_resampled'
    if args.num_category == 10:
        catfile = os.path.join(data_path, 'modelnet10_shape_names.txt')
    else:
        catfile = os.path.join(data_path, 'modelnet40_shape_names.txt')
    class_names = [line.strip() for line in open(catfile)]

    # 找到 log_dir
    if args.log_dir is None:
        experiment_dir = get_latest_log_dir()
        print(f'Auto select latest log_dir: {experiment_dir.name}')
    else:
        experiment_dir = Path(ROOT_DIR) / 'tools' / 'pointnet' / 'log' / 'classification' / args.log_dir

    # 加载模型
    classifier = load_classifier(experiment_dir, args.num_category, args.use_normals, device)

    # 判断模式
    if args.point_cloud is not None:
        predict_single(args, device, classifier, class_names)
    else:
        predict_batch(args, device, classifier, class_names)


if __name__ == '__main__':
    args = parse_args()
    main(args)
