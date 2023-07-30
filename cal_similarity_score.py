import skeleton_alignment as align
import skeleton_similarity as simi
# import utils.visualize as vis
import numpy as np
import matplotlib.pyplot as plt
import time

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
    align.align_skeletons(skeleton1, frame)

    # 根据骨架计算各个关节的角度
    skeleton1_angles = simi.qualify_skeleton(skeleton1)
    frame_angles = simi.qualify_skeleton(frame)

    # 画出对齐后的用户骨架和标准骨架，测试用
    # vis.draw_skeletons(skeleton1, frame)

    # 计算用户骨架和标准骨架的相似度
    score, jwd = simi.score_similarity(skeleton1, skeleton1_angles, frame, frame_angles)
    return score, jwd


if __name__ == '__main__':
    # 用于测试
    config = "model/td-hm_vipnas-res50_8xb64-210e_coco-256x192.py"
    checkpoint = "model/td-hm_vipnas-res50_8xb64-210e_coco-256x192-35d4bff9_20220917.pth"

    # config = "model/simcc_res50_8xb64-210e_coco-256x192.py"
    # checkpoint = "model/simcc_res50_8xb64-210e_coco-256x192-8e0f5b59_20220919.pth"

    # config = "model/simcc_vipnas-mbv3_8xb64-210e_coco-256x192.py"
    # checkpoint = "model/simcc_vipnas-mbv3_8xb64-210e_coco-256x192-719f3489_20220922.pth"

    # config = "model/td-hm_hrnet-w48_8xb32-210e_coco-256x192.py"
    # checkpoint = "model/td-hm_hrnet-w48_8xb32-210e_coco-256x192-0e67c616_20220913.pth"
    device = "cpu"
    model = init_model(
        config,
        checkpoint,
        device=device,
        cfg_options=None)
    
    frames = np.load('standardFrames/romanian_deadlift_right/romanian_deadlift_right.npy')
    # frames = np.load('standardFrames/test_video_3/test_video_3.npy')
    # frames = np.load("standardFrames/kneeling_push_ups/kneeling_push_ups.npy")
    score = cal_similarity_score(model, "14.5.png", frames[0])
    print(score)

    # start_time = time.time()
    # for i in range(500):
    #     score = cal_similarity_score(model, "14.5.png", frames[2])
    
    # print(config[6:-3])
    # print("inference time per image:", (time.time() - start_time) / 500)
