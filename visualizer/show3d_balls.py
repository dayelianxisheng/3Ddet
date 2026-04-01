"""
点云 3D 可视化工具（纯 Python + OpenCV 实现，跨平台）
支持鼠标交互旋转、缩放、颜色切换

用法:
    python show3d_balls.py --input path/to/pointcloud.txt
    python show3d_balls.py --input path/to/pointcloud.txt --color_by xyz  # 按坐标着色
"""

import numpy as np
import cv2
import sys
import os
import argparse

showsz = 800
mousex, mousey = 0.5, 0.5
zoom = 1.0
changed = True


def normalize_channel(c):
    mn, mx = c.min(), c.max()
    if mx - mn < 1e-14:
        return np.clip(c, 0.0, 255.0)
    return (c - mn) / (mx - mn) * 255.0


def onmouse(*args):
    global mousex, mousey, changed
    y = args[1]
    x = args[2]
    mousex = x / float(showsz)
    mousey = y / float(showsz)
    changed = True


def render_ball_python(img, ixyz, colors, ballradius):
    """纯 Python 实现小球渲染，替代 C 扩展的 render_ball"""
    h, w = img.shape[:2]
    r = max(ballradius, 3)  # 保证最小可见
    for i in range(ixyz.shape[0]):
        px, py = int(ixyz[i, 0]), int(ixyz[i, 1])
        if not (0 <= px < w and 0 <= py < h):
            continue
        # BGR 顺序
        color = (int(colors[0, i]), int(colors[1, i]), int(colors[2, i]))
        cv2.circle(img, (px, py), r, color, -1)  # 实心圆，无抗锯齿


def showpoints(xyz, c_gt=None, c_pred=None, waittime=0, showrot=False, magnifyBlue=0, freezerot=False,
               background=(0, 0, 0), normalizecolor=True, ballradius=4):
    """
    显示 3D 点云，支持鼠标交互

    参数:
        xyz:            点云坐标 (N, 3)
        c_gt:           真实标签颜色 (N, 3)，可选
        c_pred:         预测颜色 (N, 3)，可选
        waittime:       显示等待时间(ms)，0=无限等待
        showrot:        是否显示旋转角度文字
        magnifyBlue:    蓝色通道增强（0/1/2）
        freezerot:      冻结旋转
        background:     背景色 (B, G, R)
        normalizecolor: 是否归一化颜色到 0-255
        ballradius:     小球半径
    """
    global showsz, mousex, mousey, zoom, changed

    # 居中并缩放到窗口尺寸
    xyz = xyz - xyz.mean(axis=0)
    radius = np.sqrt((xyz ** 2).sum(axis=1)).max()
    if radius < 1e-6:
        radius = 1.0
    xyz = xyz / (radius * 2.2) * showsz
    # 调试：打印点云范围
    print(f'  xyz range: [{xyz.min():.1f}, {xyz.max():.1f}], radius={radius:.2f}')

    # 颜色通道 BGR 顺序（适配 OpenCV）
    if c_gt is None:
        c0 = np.full(len(xyz), 255, dtype='float32')  # 白色点
        c1 = np.full(len(xyz), 255, dtype='float32')
        c2 = np.full(len(xyz), 255, dtype='float32')
    else:
        c0 = c_gt[:, 0].astype('float32')
        c1 = c_gt[:, 1].astype('float32')
        c2 = c_gt[:, 2].astype('float32')

    if normalizecolor:
        c0 = normalize_channel(c0)
        c1 = normalize_channel(c1)
        c2 = normalize_channel(c2)

    # OpenCV 用 BGR，原始顺序是 RGB -> [B, G, R]
    colors = np.stack([c2, c1, c0], axis=0).astype('uint8')

    cv2.namedWindow('show3d')
    cv2.moveWindow('show3d', 0, 0)
    cv2.setMouseCallback('show3d', onmouse)

    show = np.full((showsz, showsz, 3), background, dtype='uint8')

    def render():
        global showsz, mousex, mousey, zoom, changed

        rotmat = np.eye(3)
        if not freezerot:
            xangle = (mousey - 0.5) * np.pi * 1.2
        else:
            xangle = 0
        rotmat = rotmat @ np.array([
            [1.0, 0.0, 0.0],
            [0.0, np.cos(xangle), -np.sin(xangle)],
            [0.0, np.sin(xangle), np.cos(xangle)],
        ])
        if not freezerot:
            yangle = (mousex - 0.5) * np.pi * 1.2
        else:
            yangle = 0
        rotmat = rotmat @ np.array([
            [np.cos(yangle), 0.0, -np.sin(yangle)],
            [0.0, 1.0, 0.0],
            [np.sin(yangle), 0.0, np.cos(yangle)],
        ])
        rotmat *= zoom
        nxyz = xyz @ rotmat.T + [showsz / 2, showsz / 2, 0]
        ixyz = nxyz.astype('int32')

        show[:] = background
        render_ball_python(show, ixyz, colors, ballradius)

        # 蓝色通道增强
        if magnifyBlue > 0:
            show[:, :, 0] = np.maximum(show[:, :, 0], np.roll(show[:, :, 0], 1, axis=0))
            if magnifyBlue >= 2:
                show[:, :, 0] = np.maximum(show[:, :, 0], np.roll(show[:, :, 0], -1, axis=0))
            show[:, :, 0] = np.maximum(show[:, :, 0], np.roll(show[:, :, 0], 1, axis=1))
            if magnifyBlue >= 2:
                show[:, :, 0] = np.maximum(show[:, :, 0], np.roll(show[:, :, 0], -1, axis=1))

        if showrot:
            cv2.putText(show, f'xangle: {int(xangle / np.pi * 180)}deg', (20, showsz - 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            cv2.putText(show, f'yangle: {int(yangle / np.pi * 180)}deg', (20, showsz - 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            cv2.putText(show, f'zoom: {int(zoom * 100)}%', (20, showsz - 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    changed = True
    while True:
        if changed:
            render()
            changed = False
        cv2.imshow('show3d', show)
        if waittime == 0:
            cmd = cv2.waitKey(10) & 0xFF
        else:
            cmd = cv2.waitKey(waittime) & 0xFF

        if cmd == ord('q'):
            break
        elif cmd == ord('Q'):
            sys.exit(0)

        # 切换颜色模式
        if cmd in (ord('t'), ord('p')):
            if cmd == ord('t'):
                if c_gt is None:
                    c0 = np.full(len(xyz), 255, dtype='float32')
                    c1 = np.full(len(xyz), 255, dtype='float32')
                    c2 = np.full(len(xyz), 255, dtype='float32')
                else:
                    c0 = c_gt[:, 0].astype('float32')
                    c1 = c_gt[:, 1].astype('float32')
                    c2 = c_gt[:, 2].astype('float32')
            else:
                if c_pred is None:
                    c0 = np.full(len(xyz), 255, dtype='float32')
                    c1 = np.full(len(xyz), 255, dtype='float32')
                    c2 = np.full(len(xyz), 255, dtype='float32')
                else:
                    c0 = c_pred[:, 0].astype('float32')
                    c1 = c_pred[:, 1].astype('float32')
                    c2 = c_pred[:, 2].astype('float32')

            if normalizecolor:
                c0 = normalize_channel(c0)
                c1 = normalize_channel(c1)
                c2 = normalize_channel(c2)

            colors = np.stack([c2, c1, c0], axis=0).astype('uint8')
            changed = True

        # 缩放
        if cmd == ord('n'):
            zoom *= 1.1
            changed = True
        elif cmd == ord('m'):
            zoom /= 1.1
            changed = True
        elif cmd == ord('r'):
            zoom = 1.0
            changed = True
        elif cmd == ord('s'):
            cv2.imwrite('show3d.png', show)

        if waittime != 0:
            break

    cv2.destroyAllWindows()
    return cmd


def main():
    parser = argparse.ArgumentParser(description='3D 点云可视化工具')
    parser.add_argument('--input', type=str, required=True, help='点云文件路径 (.txt, 格式: x,y,z[,nx,ny,nz])')
    parser.add_argument('--npoints', type=int, default=1024, help='采样点数')
    parser.add_argument('--ballradius', type=int, default=8, help='小球半径')
    parser.add_argument('--color_by', type=str, default=None,
                        choices=['xyz', 'none'],
                        help='着色方式: xyz=按坐标归一化着色, none=白色')
    parser.add_argument('--background', type=str, default='black',
                        choices=['black', 'white'], help='背景色')
    opt = parser.parse_args()

    # 加载点云
    if not os.path.exists(opt.input):
        print(f'文件不存在: {opt.input}')
        sys.exit(1)

    points = np.loadtxt(opt.input, delimiter=',').astype(np.float32)
    print(f'Loaded: {points.shape}, range: [{points.min():.2f}, {points.max():.2f}]')

    xyz = points[:, 0:3]

    # 采样
    if xyz.shape[0] > opt.npoints:
        choice = np.random.choice(xyz.shape[0], opt.npoints, replace=False)
        xyz = xyz[choice]

    # 着色
    c_gt = None
    if opt.color_by == 'xyz':
        # 按 xyz 坐标归一化到 [0,1] 作为 RGB 颜色
        xyz_norm = (xyz - xyz.min(axis=0)) / (xyz.max(axis=0) - xyz.min(axis=0) + 1e-14)
        c_gt = xyz_norm * 255.0

    bg = (255, 255, 255) if opt.background == 'white' else (0, 0, 0)

    showpoints(xyz, c_gt=c_gt, waittime=0, ballradius=opt.ballradius,
               background=bg, normalizecolor=True)


if __name__ == '__main__':
    main()
