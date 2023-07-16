import requests

data = {'name': 'Alice'}

startMsg = {'msg': 'START',
            'target': 'waaaaao'}

requests.post('http://localhost:9001/Start/', json = startMsg)

#这下面是测试的照片
files = {'img': open('user_keyframe1.png', 'rb')}
requests.post('http://localhost:9001/Img/', files= files)

respone = requests.post('http://localhost:9001/End/')
result = respone.json()

print("recv" + str( result['score']))
#response = requests.post('http://localhost:5000/hello', json=data)
#result = response.json()

#message = result['message']
#print(message)