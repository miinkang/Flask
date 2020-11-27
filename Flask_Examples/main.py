import flask
from flask import Flask, request, render_template
# from sklearn.externals import joblib
import joblib

from sklearn import externals
import numpy as np
from scipy import misc

import tensorflow as tf
import requests
from PIL import Image

from io import BytesIO
# import os


app = Flask(__name__)


# 메인 페이지 라우팅
@app.route("/")
@app.route("/index")
def index():
    return flask.render_template('index.html')


# 데이터 예측 처리
@app.route('/predict', methods=['POST'])
def make_prediction():
    if request.method == 'POST':

        # 업로드 파일 처리 분기

        image_url = request.values['image']



        response = requests.get(image_url)
        image = Image.open(BytesIO(response.content)).convert('RGB')

        # image = Image.open(image_url).convert('RGB')
        image = image.resize((200, 200))
        image = np.array(image)
        image = image.reshape((200, 200, 3))

        # file = request.files['image']
        # if not file: return render_template('index.html', label="No Files")
        #
        # # 이미지 픽셀 정보 읽기
        # # 알파 채널 값 제거 후 1차원 Reshape
        # img = misc.imread(file)
        # img = img[:, :, :3]
        # img = img.reshape(200,200,3)
        # img = img.reshape(1, -1)

        # 입력 받은 이미지 예측
        is_korea_flag = True

        # os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

        # 예측 값을 1차원 배열로부터 확인 가능한 문자열로 변환
        # plt.imshow(image)
        x_test = [image]
        x_test = np.array(x_test)
        x_test = x_test / 255

        pred = model.predict(x_test)
        if pred < 0.5:
            is_korea_flag = True
        else:
            is_korea_flag = False


        # 결과 리턴
        return render_template('index.html', label=is_korea_flag)


if __name__ == '__main__':
    # 모델 로드
    # ml/model.py 선 실행 후 생성
    model = tf.keras.models.load_model('./model/xception.h5')
    # model.load_weights('./model/xception.h5')
    # Flask 서비스 스타트
    app.run(host='127.0.0.1', port=8000, debug=True)
