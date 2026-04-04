import argparse
from pathlib import Path

import numpy as np


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = ROOT_DIR / 'data' / 'stanford_indoor3d'
CLASSES = [
    'ceiling',
    'floor',
    'wall',
    'beam',
    'column',
    'window',
    'door',
    'table',
    'chair',
    'sofa',
    'bookcase',
    'board',
    'clutter',
]
CLASS_TO_LABEL = {name: idx for idx, name in enumerate(CLASSES)}
FALLBACK_LABEL = CLASS_TO_LABEL['clutter']


def parse_args():
    parser = argparse.ArgumentParser('Prepare S3DIS room .npy files for semantic segmentation training.')
    parser.add_argument(
        '--raw_root',
        type=str,
        required=True,
        help='Path to the raw Stanford3dDataset_v1.2_Aligned_Version directory',
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help='Directory to store preprocessed Area_*.npy room files',
    )
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing room .npy files')
    return parser.parse_args()


def resolve_path(path_text):
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = (ROOT_DIR / path).resolve()
    return path


def iter_room_annotations(raw_root):
    for area_dir in sorted(raw_root.glob('Area_*')):
        if not area_dir.is_dir():
            continue
        for room_dir in sorted(area_dir.iterdir()):
            annotations_dir = room_dir / 'Annotations'
            if annotations_dir.is_dir():
                yield area_dir.name, room_dir.name, annotations_dir


def load_instance_points(txt_path):
    rows = []
    bad_rows = 0
    with txt_path.open('r', encoding='utf-8', errors='ignore') as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            stripped = raw_line.strip()
            if not stripped:
                continue

            parts = stripped.split()
            if len(parts) < 6:
                bad_rows += 1
                continue

            try:
                rows.append([float(value) for value in parts[:6]])
            except ValueError:
                bad_rows += 1

    if not rows:
        raise ValueError(f'No valid XYZRGB rows found in {txt_path}')

    if bad_rows:
        print(f'Warning: skipped {bad_rows} malformed rows in {txt_path}')

    return np.asarray(rows, dtype=np.float32)


def collect_room_points(annotations_dir):
    room_parts = []
    for txt_path in sorted(annotations_dir.glob('*.txt')):
        class_name = txt_path.stem.split('_')[0].lower()
        label = CLASS_TO_LABEL.get(class_name, FALLBACK_LABEL)
        points = load_instance_points(txt_path)
        labels = np.full((points.shape[0], 1), label, dtype=np.float32)
        room_parts.append(np.concatenate([points[:, :6], labels], axis=1))
    if not room_parts:
        raise FileNotFoundError(f'No annotation .txt files found in {annotations_dir}')

    room_data = np.concatenate(room_parts, axis=0).astype(np.float32)
    room_data[:, :3] -= np.amin(room_data[:, :3], axis=0)
    return room_data


def main():
    args = parse_args()
    raw_root = resolve_path(args.raw_root)
    output_dir = resolve_path(args.output_dir)

    if not raw_root.is_dir():
        raise FileNotFoundError(f'raw_root does not exist: {raw_root}')

    rooms = list(iter_room_annotations(raw_root))
    if not rooms:
        raise FileNotFoundError(
            f'No S3DIS room annotations were found under {raw_root}. Expected Area_x/room_name/Annotations/*.txt'
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f'Preparing {len(rooms)} rooms from {raw_root} -> {output_dir}')

    converted = 0
    skipped = 0
    for area_name, room_name, annotations_dir in rooms:
        out_path = output_dir / f'{area_name}_{room_name}.npy'
        if out_path.exists() and not args.overwrite:
            skipped += 1
            print(f'Skip existing {out_path.name}')
            continue

        room_data = collect_room_points(annotations_dir)
        np.save(out_path, room_data)
        converted += 1
        print(f'Wrote {out_path.name}: {room_data.shape[0]} points')

    print(f'Finished. Converted {converted} rooms, skipped {skipped}.')


if __name__ == '__main__':
    main()
