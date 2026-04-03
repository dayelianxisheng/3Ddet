import numpy as np

def normalize_data(batch_data):
    """
    归一化批次点云数据，以原点云中心为原点进行缩放
    输入:
        BxNxC 数组
    输出:
        BxNxC 归一化后的数组
    """
    B, N, C = batch_data.shape
    normal_data = np.zeros((B, N, C))
    for b in range(B):
        pc = batch_data[b]
        centroid = np.mean(pc, axis=0)  # 计算中心点
        pc = pc - centroid               # 平移到原点
        m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))  # 最大距离
        pc = pc / m                      # 归一化
        normal_data[b] = pc
    return normal_data


def shuffle_data(data, labels):
    """
    打乱数据和标签
    输入:
        data: BxNx... numpy 数组
        label: Bx... numpy 数组
    返回:
        打乱后的 data, label 和 shuffle 后的索引
    """
    idx = np.arange(len(labels))
    np.random.shuffle(idx)
    return data[idx, ...], labels[idx], idx


def shuffle_points(batch_data):
    """
    打乱每个点云中点的顺序
    整个 batch 使用相同的打乱索引
    输入:
        BxNxC 数组
    输出:
        BxNxC 打乱后的数组
    """
    idx = np.arange(batch_data.shape[1])
    np.random.shuffle(idx)
    return batch_data[:, idx, :]


def rotate_point_cloud(batch_data):
    """
    随机旋转点云进行数据增强
    绕 Z 轴（竖直向上）方向旋转，每个形状独立旋转
    输入:
        BxNx3 数组，点云批次
    输出:
        BxNx3 数组，旋转后的点云批次
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
    return rotated_data


def rotate_point_cloud_z(batch_data):
    """
    随机旋转点云进行数据增强
    绕 Z 轴旋转，旋转角为 [0, 2π] 随机值
    输入:
        BxNx3 数组，点云批次
    输出:
        BxNx3 数组，旋转后的点云批次
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, sinval, 0],
                                    [-sinval, cosval, 0],
                                    [0, 0, 1]])
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
    return rotated_data


def rotate_point_cloud_with_normal(batch_xyz_normal):
    """
    随机旋转带有法向量的点云 XYZ 和 normal 一起旋转
    输入:
        batch_xyz_normal: BxNx6 数组，前3通道为 XYZ，后3通道为法向量
    输出:
        BxNx6 数组，旋转后的 XYZ 和法向量
    """
    for k in range(batch_xyz_normal.shape[0]):
        rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])
        shape_pc = batch_xyz_normal[k, :, 0:3]
        shape_normal = batch_xyz_normal[k, :, 3:6]
        batch_xyz_normal[k, :, 0:3] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
        batch_xyz_normal[k, :, 3:6] = np.dot(shape_normal.reshape((-1, 3)), rotation_matrix)
    return batch_xyz_normal


def rotate_perturbation_point_cloud_with_normal(batch_data, angle_sigma=0.06, angle_clip=0.18):
    """
    通过小角度旋转对点云进行扰动数据增强
    输入:
        BxNx6 数组，点云及法向量
    输出:
        BxNx6 数组，扰动后的点云及法向量
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        # 生成随机旋转角，带截断的高斯分布
        angles = np.clip(angle_sigma * np.random.randn(3), -angle_clip, angle_clip)
        Rx = np.array([[1, 0, 0],
                       [0, np.cos(angles[0]), -np.sin(angles[0])],
                       [0, np.sin(angles[0]), np.cos(angles[0])]])
        Ry = np.array([[np.cos(angles[1]), 0, np.sin(angles[1])],
                       [0, 1, 0],
                       [-np.sin(angles[1]), 0, np.cos(angles[1])]])
        Rz = np.array([[np.cos(angles[2]), -np.sin(angles[2]), 0],
                       [np.sin(angles[2]), np.cos(angles[2]), 0],
                       [0, 0, 1]])
        R = np.dot(Rz, np.dot(Ry, Rx))
        shape_pc = batch_data[k, :, 0:3]
        shape_normal = batch_data[k, :, 3:6]
        rotated_data[k, :, 0:3] = np.dot(shape_pc.reshape((-1, 3)), R)
        rotated_data[k, :, 3:6] = np.dot(shape_normal.reshape((-1, 3)), R)
    return rotated_data


def rotate_point_cloud_by_angle(batch_data, rotation_angle):
    """
    按指定角度绕 Z 轴旋转点云
    输入:
        BxNx3 数组，点云批次
        rotation_angle: 旋转角度（弧度）
    输出:
        BxNx3 数组，旋转后的点云批次
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])
        shape_pc = batch_data[k, :, 0:3]
        rotated_data[k, :, 0:3] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
    return rotated_data


def rotate_point_cloud_by_angle_with_normal(batch_data, rotation_angle):
    """
    按指定角度绕 Z 轴旋转点云（包含法向量）
    输入:
        BxNx6 数组，点云及法向量批次
        rotation_angle: 旋转角度（弧度）
    输出:
        BxNx6 数组，旋转后的点云及法向量批次
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])
        shape_pc = batch_data[k, :, 0:3]
        shape_normal = batch_data[k, :, 3:6]
        rotated_data[k, :, 0:3] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
        rotated_data[k, :, 3:6] = np.dot(shape_normal.reshape((-1, 3)), rotation_matrix)
    return rotated_data


def rotate_perturbation_point_cloud(batch_data, angle_sigma=0.06, angle_clip=0.18):
    """
    通过小角度旋转对点云进行扰动数据增强
    输入:
        BxNx3 数组，点云批次
    输出:
        BxNx3 数组，扰动后的点云批次
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        angles = np.clip(angle_sigma * np.random.randn(3), -angle_clip, angle_clip)
        Rx = np.array([[1, 0, 0],
                       [0, np.cos(angles[0]), -np.sin(angles[0])],
                       [0, np.sin(angles[0]), np.cos(angles[0])]])
        Ry = np.array([[np.cos(angles[1]), 0, np.sin(angles[1])],
                       [0, 1, 0],
                       [-np.sin(angles[1]), 0, np.cos(angles[1])]])
        Rz = np.array([[np.cos(angles[2]), -np.sin(angles[2]), 0],
                       [np.sin(angles[2]), np.cos(angles[2]), 0],
                       [0, 0, 1]])
        R = np.dot(Rz, np.dot(Ry, Rx))
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), R)
    return rotated_data


def jitter_point_cloud(batch_data, sigma=0.01, clip=0.05):
    """
    随机抖动点云中的每个点
    输入:
        BxNx3 数组，点云批次
    输出:
        BxNx3 数组，抖动后的点云批次
    """
    B, N, C = batch_data.shape
    assert(clip > 0)
    jittered_data = np.clip(sigma * np.random.randn(B, N, C), -1 * clip, clip)
    jittered_data += batch_data
    return jittered_data


def shift_point_cloud(batch_data, shift_range=0.1):
    """
    随机平移点云
    输入:
        BxNx3 数组，点云批次
    输出:
        BxNx3 数组，平移后的点云批次
    """
    B, N, C = batch_data.shape
    shifts = np.random.uniform(-shift_range, shift_range, (B, 3))
    for batch_index in range(B):
        batch_data[batch_index, :, :] += shifts[batch_index, :]
    return batch_data


def random_scale_point_cloud(batch_data, scale_low=0.8, scale_high=1.25):
    """
    随机缩放点云
    输入:
        BxNx3 数组，点云批次
    输出:
        BxNx3 数组，缩放后的点云批次
    """
    B, N, C = batch_data.shape
    scales = np.random.uniform(scale_low, scale_high, B)
    for batch_index in range(B):
        batch_data[batch_index, :, :] *= scales[batch_index]
    return batch_data


def random_point_dropout(batch_pc, max_dropout_ratio=0.875):
    """
    随机丢弃点云中的点
    输入:
        batch_pc: BxNx3 点云批次
    输出:
        BxNx3 丢弃后的点云批次
    """
    for b in range(batch_pc.shape[0]):
        dropout_ratio = np.random.random() * max_dropout_ratio  # 0 ~ max_dropout_ratio
        drop_idx = np.where(np.random.random((batch_pc.shape[1])) <= dropout_ratio)[0]
        if len(drop_idx) > 0:
            batch_pc[b, drop_idx, :] = batch_pc[b, 0, :]  # 用第一个点填充
    return batch_pc
