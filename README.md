# PostureScoring-BackEnd

This is the back-end of the Automatic Scoring System.

Using **FlaskServer** and **MMPose** as the upstream inference model on **python 3.8**.

### Environment

```shell
pip install Flask
pip install Flask-Cors
pip install opencv-python

# To install MMPose:
conda install pytorch torchvision -c pytorch

# 安装 MMEngine 和 MMCV:
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.1"

# 从源码安装MMPose
git clone https://github.com/open-mmlab/mmpose.git
cd mmpose
pip install -r requirements.txt
pip install -v -e .
# "-v" 表示输出更多安装相关的信息
# "-e" 表示以可编辑形式安装，这样可以在不重新安装的情况下，让本地修改直接生效

```

### Start Server

```shell
python Server.py
```

