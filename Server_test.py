from flask import Flask,jsonify,request,Response,url_for
from flask_cors import *
import cv2
import numpy as np
import json
from func import *

app = Flask(__name__)
CORS(app,supports_credentials = True)

context_rcd = {
    'imgNum':0
} 

@app.route('/sendStart/',methods=['POST'])
def sendStart():
    print(request)
    context_rcd['imgNum'] = 0
    return "Start!"

@app.route('/sendEnd/',methods=['POST'])
def sendEnd():
    print(request)
    res = calc(context_rcd['imgNum'])
    return "End with {} frames in all! {}.0/10.0!".format(context_rcd['imgNum'], res)

@app.route('/sendImgs/',methods=['POST'])
def images():
    # if request.method == 'POST':
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

# @app.route('/get_image', methods=['GET'])
# def get_image():
#     image = cv2.imread('frame.jpeg')
#     imgbytes = cv2.imencode(".jpeg", image)[1].tobytes()
#     return Response(imgbytes, mimetype="image/jpeg")

@app.route('/')
def index():
    server_ip = request.remote_addr
    return "Server IP: {}".format(server_ip) 

if __name__ == '__main__':
    app.run(host='0.0.0.0',debug=True,port=5000)
