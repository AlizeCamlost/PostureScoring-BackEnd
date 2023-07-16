import test_scale_skeleton as tss
import skeleton_similarity as simi
import numpy as np

from mmpose.apis import inference_topdown, init_model
from mmpose.structures import merge_data_samples

# 输入：
# model在服务器初始化
# img_path，用户的单张图片在服务器上的路径
# frame，已经匹配好的一帧标准骨架，shape为(17,2)
# 输出：
# score，用户骨架和标准骨架的相似度
def cal_similarity_score(model, img_path, frame):

    # 用模型提取用户骨架
    batch_results = inference_topdown(model, img_path)
    results = merge_data_samples(batch_results)
    skeleton1 = results.pred_instances.keypoints[0]

    # 对齐用户骨架和标准骨架
    tss.align_skeletons(skeleton1, frame)

    # 根据骨架计算各个关节的角度
    skeleton1_angles = simi.qualify_skeleton(skeleton1)
    frame_angles = simi.qualify_skeleton(frame)

    # 计算用户骨架和标准骨架的相似度
    score = simi.score_similarity(skeleton1, skeleton1_angles, frame, frame_angles)
    return score

if __name__ == '__main__':
    # 用于测试

    config = "demo/simcc_vipnas-mbv3_8xb64-210e_coco-256x192.py"
    checkpoint = "demo/simcc_vipnas-mbv3_8xb64-210e_coco-256x192-719f3489_20220922.pth"
    device = "cpu"
    model = init_model(
        config,
        checkpoint,
        device=device,
        cfg_options=None)
    
    frames = np.load('demo/standard.npy')

    score = cal_similarity_score(model, "tests/data/coco/Tpose_standard.jpeg", frames[0])
    print(score)
