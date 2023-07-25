from flask import Flask,jsonify,request,Response,url_for
from flask_cors import *
import cv2
import numpy as np
import json
import os

import cal_similarity_score as cal
from mmpose.apis import init_model

app = Flask(__name__)
CORS(app,supports_credentials = True)

# test by sbf
context_rcd = {
    'imgNum':0
} 

@app.route('/')
def index():
    server_ip = request.remote_addr
    return "Server IP: {}".format(server_ip) 


@app.route('/sendStart/',methods=['POST'])  # send the "START" flag
def sendStart():
    print(request)
    context_rcd['imgNum'] = 0
    return "Start!"

@app.route('/sendEnd/',methods=['POST'])    # send the "END" flag
def sendEnd():
    print(request)
    # res = calc(context_rcd['imgNum'])
    res = 5
    return "End with {} frames in all! {}.0/10.0!".format(context_rcd['imgNum'], res)

@app.route('/sendImgs/',methods=['POST'])   # send a picture
def images():
    print('receive')
    print(request)
    img_dict = request.files.to_dict()
    print(img_dict)
    dict_len = len(img_dict)
    print("dictLen",dict_len)
    for key in img_dict:
        tmpImg = cv2.imdecode(np.asarray(bytearray(img_dict[key].read()),dtype='uint8'), cv2.IMREAD_COLOR)
        cv2.imwrite("{}.jpg".format(key), tmpImg)
        context_rcd['imgNum'] += 1
    # return Response()
    return "10.0/10.0! {} frame(s) in all.".format(dict_len)

#上面是施博凡之前的test，不用管。这下面就不是test了

@app.route('/Start/', methods = ['POST'])
def startHandler():
    addr = request.remote_addr
    msg = request.json

    clientDataDict[addr] = ClientData(msg['target'])
    clientScoreDict[addr] = ClientScoreData(addr)
    print("addr: " + addr + " Recv.")
    print('target: ' + clientDataDict[addr].targetAction)
    return Response()

@app.route('/Img/', methods = ['POST'])
def imgHandler():
    addr = request.remote_addr
    # file = request.files.to_dict()
    img_dict = request.files.to_dict()
    for key in img_dict:  # only one key-value pair
        tmpImg = cv2.imdecode(np.asarray(bytearray(img_dict[key].read()),dtype='uint8'), cv2.IMREAD_COLOR)
        # cv2.imwrite("{}.jpg".format(key), tmpImg)
        # context_rcd['imgNum'] += 1
        clientDataDict[addr].addImg(tmpImg)

        # img =cv2.imdecode(np.asarray(bytearray(file['img'].read()),dtype='uint8'), cv2.IMREAD_COLOR)
        # clientDataDict[addr].addImg(img)

        #cv2.imwrite('img2.jpg', img) #这句话是测试用的

        # 找到对应的标准动作帧
        target = clientDataDict[addr].targetAction
        standardFrame = standardData[target]
    
        # 下面调计算score的函数，返回值score，joint_wise_distance
        score, _ = cal.cal_similarity_score(model, tmpImg, standardFrame[int(key)]) # 帧号要从请求里解析
        print("score:",score)

    #计算完以后算一个分给到我的dict里面
        clientScoreDict[addr].addScore(score)

    return Response()

@app.route('/End/', methods = ['POST'])
def endHandler():
    addr = request.remote_addr
    ret = clientScoreDict[addr].avg()
    #ret = 100
    # return {'score': str(ret)}
    return "score:{}".format(ret)


if __name__ == '__main__':

    def readStandardData( dict ):
        #像这样读就行 xxx是动作的名字
        st = np.load('standard.npy')
        dict['pushup'] = st

    def readStandardFrames(dict, folderPath): #'./standardFrames',但帧的存储格式是./standardFrames/xxx/xxx.npy
        for root, dirs, files in os.walk(folderPath):
            for file in files:
                if file.endswith('.npy'):
                    filePath = os.path.join(root, file)
                    standardFrames = np.load(filePath)
                    dict[file[:-4]] = standardFrames
        
        # # 打印加载的.npy文件
        # for file, data in dict.items():
        #     print(f'文件名: {file}')
        #     print(f'数据维度: {data.shape}')  # 可选
        #     print(f'数据内容: {data}')  # 可选
        #     print()

    class ClientData:
        imgList = []
        targetAction = ""

        def __init__(self, tag):
            self.targetAction = tag

        def addImg(self, newImg):
            self.imgList.append(newImg)

    class ClientScoreData:
        addr=""
        scoreList = []

        def __init__(self, naddr):
            self.addr = naddr

        def addScore(self, nscore):
            self.scoreList.append(nscore)

        def avg(self):
            cur = 0.0
            for score in self.scoreList:
                cur += score

            return cur / len(self.scoreList)

    config = "model/simcc_vipnas-mbv3_8xb64-210e_coco-256x192.py"
    checkpoint = "model/simcc_vipnas-mbv3_8xb64-210e_coco-256x192-719f3489_20220922.pth"
    # config = "model/simcc_res50_8xb64-210e_coco-256x192.py"
    # checkpoint = "model/simcc_res50_8xb64-210e_coco-256x192-8e0f5b59_20220919.pth"
    device = "cpu"
    model = init_model(
        config,
        checkpoint,
        device=device,
        cfg_options=None)


    clientDataDict = {}
    clientScoreDict = {}
    standardData = {}
    # readStandardData(standardData)
    readStandardFrames(standardData, './standardFrames')

    app.run(host='0.0.0.0',debug=True,port=8080)   # NOTE: port-9001 is to client.py, port-5000 is to react-app

