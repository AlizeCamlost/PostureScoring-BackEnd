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

    input_centroid = np.mean(skeleton1, axis=0)
    standard_centroid = np.mean(skeleton2, axis=0)

    # 去中心点
    source_centered = skeleton1 - input_centroid
    target_centered = skeleton2 - standard_centroid

    # 协方差矩阵
    covariance_matrix = np.dot(source_centered.T, target_centered)

    # SVD
    U, _, Vt = np.linalg.svd(covariance_matrix)
    rotation_matrix = np.matmul(U,Vt)
    #R, s = orthogonal_procrustes(source_centered, target_centered)

    # 是否需要进行镜像翻转
    if np.linalg.det(rotation_matrix) < 0:
        rotation_matrix[:, -1] *= -1

    scale_factor = np.sqrt(np.sum(target_centered ** 2) / np.sum(source_centered ** 2))

    skeleton1[:] = np.dot(source_centered * scale_factor, rotation_matrix) + standard_centroid

#不用管
def acc_scale_skeletons(skeleton1, skeleton2):
    diff = np.linalg.norm(skeleton1 - skeleton2) / 17
    return diff
