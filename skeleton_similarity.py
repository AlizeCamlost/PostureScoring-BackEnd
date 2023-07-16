import numpy as np

def cal_angle(point1, point2, point3):
    vector1 = point1 - point2
    vector2 = point3 - point2
    cos = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
    angle = np.arccos(cos)
    angle = angle * 180 / np.pi
    return angle

# input: skeleton
# output: angle = [a1, a2, ..., a8]
# calculate the histogram of the angle between the limbs
def qualify_skeleton(skeleton):
    a1 = cal_angle(skeleton[10], skeleton[8], skeleton[6])
    a2 = cal_angle(skeleton[8], skeleton[6], skeleton[12])
    a3 = cal_angle(skeleton[6], skeleton[12], skeleton[14])
    a4 = cal_angle(skeleton[12], skeleton[14], skeleton[16])

    a5 = cal_angle(skeleton[11], skeleton[13], skeleton[15])
    a6 = cal_angle(skeleton[5], skeleton[11], skeleton[13])
    a7 = cal_angle(skeleton[7], skeleton[5], skeleton[11])
    a8 = cal_angle(skeleton[9], skeleton[7], skeleton[5])

    angle = [a1, a2, a3, a4, a5, a6, a7, a8]
    return angle

# input: angle1--skeleton1, angle2--skeleton2
def cal_angle_cos_similarity(angle1, angle2):
    angle1 = np.array(angle1)
    angle2 = np.array(angle2)
    cos = np.dot(angle1, angle2) / (np.linalg.norm(angle1) * np.linalg.norm(angle2))
    return cos
    # return (cos + 1) / 2 * 100

def cal_angle_distance(angle1, angle2):
    angle1 = np.array(angle1)
    angle2 = np.array(angle2)
    # weight = np.array([1, 1, 1, 1, 1, 1, 1, 1])
    # diff = np.linalg.norm(angle1 - angle2, axis=1)
    diff = np.abs(angle1 - angle2)
    return diff

def score_angle_similarity(angle1, angle2):
    diff = joint_wise_distance(angle1, angle2)
    weight = np.ones(8)
    weight[1] = weight[2] = weight[5] = weight[6] = 2
    weight = weight / np.sum(weight)
    diff = diff / 180
    diff = 1 - diff
    diff = diff * weight
    similarity = np.sum(diff) * 100
    return similarity

def joint_wise_distance(angle1, angle2):
    angle1 = np.array(angle1)
    angle2 = np.array(angle2)
    diff = np.abs(angle1 - angle2)
    return diff

def point_wise_distance(skeleton1, skeleton2):
    diff = np.linalg.norm(skeleton1 - skeleton2, axis=1)
    return diff

def score_point_similarity(skeleton1, skeleton2):
    diff = point_wise_distance(skeleton1, skeleton2)
    max_diff = np.max(diff)
    weights = np.ones(17)
    weights[0:5] = 0
    weights[7] = weights[8] = weights[13] = weights[14] = 2
    weights = weights / np.sum(weights)
    diff = diff * weights
    # similarity = 100 - np.sum(diff) / (np.linalg.norm(skeleton1[6] - skeleton1[5]) * 17) * 100
    similarity = 100 - np.sum(diff) / (max_diff * 17) * 100
    return similarity



#只看这个就行了，目前调用的是这个
def score_similarity(skeleton1, angle1, skeleton2, angle2):
    angle_similarity = score_angle_similarity(angle1, angle2)
    point_similarity = score_point_similarity(skeleton1, skeleton2)
    similarity = 0.7*angle_similarity + 0.3*point_similarity
    return similarity