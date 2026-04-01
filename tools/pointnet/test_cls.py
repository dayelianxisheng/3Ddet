from dataset.ModelNetDataLoader import ModelNetDataLoader
import argparse
import numpy as np
import os
import torch
import logging
from tqdm import tqdm
import sys
import importlib
from pathlib import Path

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(ROOT_DIR, 'model'))


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('Testing')
    parser.add_argument('--use_cpu', action='store_true', default=False, help='use cpu mode')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--batch_size', type=int, default=24, help='batch size in training')
    parser.add_argument('--num_category', default=40, type=int, choices=[10, 40], help='training on ModelNet10/40')
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number')
    parser.add_argument('--log_dir', type=str, default=None, help='Experiment root (default: latest)')
    parser.add_argument('--use_normals', action='store_true', default=False, help='use normals')
    parser.add_argument('--use_uniform_sample', action='store_true', default=False, help='use uniform sampling')
    parser.add_argument('--num_votes', type=int, default=3, help='Aggregate classification scores with voting')
    return parser.parse_args()


def test(model, loader, device, num_class=40, vote_num=1):
    mean_correct = []
    class_acc = np.zeros((num_class, 3))
    classifier = model.eval()
    vote_pool = torch.zeros(vote_num, device=device, requires_grad=False)

    for j, (points, target) in tqdm(enumerate(loader), total=len(loader)):
        points, target = points.to(device), target.to(device)
        points = points.transpose(2, 1)
        pred_pool = torch.zeros(target.size()[0], num_class, device=device)

        for _ in range(vote_num):
            pred, _ = classifier(points)
            pred_pool += pred
        pred = pred_pool / vote_num
        pred_choice = pred.data.max(1)[1]

        for cat in np.unique(target.cpu()):
            classacc = pred_choice[target == cat].eq(target[target == cat].long().data).cpu().sum()
            class_acc[cat, 0] += classacc.item() / float(points[target == cat].size()[0])
            class_acc[cat, 1] += 1
        correct = pred_choice.eq(target.long().data).cpu().sum()
        mean_correct.append(correct.item() / float(points.size()[0]))

    class_acc[:, 2] = class_acc[:, 0] / class_acc[:, 1]
    class_acc = np.mean(class_acc[:, 2])
    instance_acc = np.mean(mean_correct)
    return instance_acc, class_acc


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() and not args.use_cpu else 'cpu')

    '''FIND LOG DIR'''
    if args.log_dir is None:
        exp_base = Path(ROOT_DIR) / 'tools' / 'pointnet' / 'log' / 'classification'
        if not exp_base.exists():
            raise FileNotFoundError(f'log directory not found: {exp_base}')
        candidates = sorted([d for d in exp_base.iterdir() if d.is_dir()], key=lambda d: d.stat().st_mtime)
        if not candidates:
            raise FileNotFoundError(f'No log directory found in {exp_base}')
        experiment_dir = candidates[-1]
        print(f'Auto select latest log_dir: {experiment_dir.name}')
    else:
        experiment_dir = Path(ROOT_DIR) / 'tools' / 'pointnet' / 'log' / 'classification' / args.log_dir

    '''LOG'''
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/eval.txt' % experiment_dir)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    '''DATA LOADING'''
    log_string('Load dataset ...')
    data_path = r'D:\resource\data\3d\modelnet40_normal_resampled'

    test_dataset = ModelNetDataLoader(root=data_path, args=args, split='test', process_data=False)
    testDataLoader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=10)

    '''MODEL LOADING'''
    num_class = args.num_category
    logs_dir = experiment_dir / 'logs'
    model_files = list(logs_dir.glob('*.txt'))
    if not model_files:
        raise FileNotFoundError(f'No log file found in {logs_dir}')
    # 从日志文件名推断模型名，e.g. "pointnet_cls.txt" -> "pointnet_cls"
    model_name = model_files[0].stem
    model = importlib.import_module(model_name)

    classifier = model.get_model(num_class, normal_channel=args.use_normals)
    classifier = classifier.to(device)

    checkpoint = torch.load(str(experiment_dir / 'checkpoints' / 'best_model.pth'), map_location=device, weights_only=False)
    classifier.load_state_dict(checkpoint['model_state_dict'])
    log_string(f"Loaded checkpoint from epoch {checkpoint['epoch']}")

    with torch.no_grad():
        instance_acc, class_acc = test(classifier.eval(), testDataLoader, device, vote_num=args.num_votes, num_class=num_class)
        log_string('Test Instance Accuracy: %f, Class Accuracy: %f' % (instance_acc, class_acc))


if __name__ == '__main__':
    args = parse_args()
    main(args)
