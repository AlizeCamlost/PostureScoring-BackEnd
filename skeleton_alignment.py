import numpy as np

# skeleton1: user skeleton 
# skeleton2: standard skeleton
def cal_scale_factor(skeleton1, skeleton2):
    shoulder1 = np.linalg.norm(skeleton1[6] - skeleton1[5])
    shoulder2 = np.linalg.norm(skeleton2[6] - skeleton2[5])

    k = shoulder2 / shoulder1

    return k

def scale_skeleton(skeleton, k):
    center = np.mean(skeleton, axis=0)
    # center = (skeleton[6] + skeleton[5]) / 2
    skeleton -= center
    skeleton *= k
    skeleton += center


# 根据skeleton2的尺寸调整skeleton1的尺寸并将skeleton1的中心点与skeleton2的中心点对齐
def align_skeletons_deprecated(skeleton1, skeleton2):
    k = cal_scale_factor(skeleton1, skeleton2)
    scale_skeleton(skeleton1, k)

    position_offset = skeleton2 - skeleton1
    position_difference = np.mean(position_offset, axis=0)

    skeleton1 += position_difference


def align_skeletons(skeleton1, skeleton2):
    source_centroid = np.mean(skeleton1, axis=0)
    target_centroid = np.mean(skeleton2, axis=0)

    source_centered = skeleton1 - source_centroid
    target_centered = skeleton2 - target_centroid

    covariance_matrix = np.dot(source_centered.T, target_centered)

    # 使用SVD对协方差矩阵进行分解，得到旋转矩阵
    U, _, Vt = np.linalg.svd(covariance_matrix)
    rotation_matrix = np.dot(U, Vt)

    # 计算缩放因子
    scale_factor = np.sqrt(np.sum(target_centered ** 2) / np.sum(source_centered ** 2))

    # 计算平移向量
    translation_vector = target_centroid - source_centroid * scale_factor

    # 将原始关键点集合应用缩放、旋转和平移变换
    skeleton1[:] *= scale_factor
    skeleton1[:] = np.dot(skeleton1, rotation_matrix.T)
    skeleton1[:] += translation_vector

#不用管
def acc_scale_skeletons(skeleton1, skeleton2):
    diff = np.linalg.norm(skeleton1 - skeleton2) / 17
    return diff