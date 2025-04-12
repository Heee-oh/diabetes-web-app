import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
import os

from flask import Flask, render_template, request
from dotenv import load_dotenv

from flask_bootstrap import Bootstrap5
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField
from wtforms.validators import DataRequired

# Flask 애플리케이션 초기화 및 시크릿 키 설정
app = Flask(__name__)
app.config['SECRET_KEY'] = 'hard to guess string'

# Bootstrap5 적용
bootstrap5 = Bootstrap5(app)

# 사용자 입력을 위한 폼 정의
class LabForm(FlaskForm):
    preg = StringField('# Pregnancies', validators=[DataRequired()])
    glucose = StringField('Glucose', validators=[DataRequired()])
    blood = StringField('Blood pressure', validators=[DataRequired()])
    skin = StringField('Skin thickness', validators=[DataRequired()])
    insulin = StringField('Insulin', validators=[DataRequired()])
    bmi = StringField('BMI', validators=[DataRequired()])
    dpf = StringField('DPF Score', validators=[DataRequired()])
    age = StringField('Age', validators=[DataRequired()])
    submit = SubmitField('Submit')

# 루트 및 인덱스 페이지 렌더링
@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')

# 예측 페이지: 사용자 입력 처리 및 결과 예측
@app.route('/prediction', methods=['GET', 'POST'])
def lab():
    form = LabForm()
    if form.validate_on_submit():
        # 입력값을 numpy 배열로 변환
        X_test = np.array([[
            float(form.preg.data),
            float(form.glucose.data),
            float(form.blood.data),
            float(form.skin.data),
            float(form.insulin.data),
            float(form.bmi.data),
            float(form.dpf.data),
            float(form.age.data)
        ]])

        # 학습 데이터 불러오기
        data = pd.read_csv('./diabetes.csv', sep=',')
        X = data.values[:, 0:8]
        y = data.values[:, 8]

        # MinMax 스케일링 수행
        scaler = MinMaxScaler()
        scaler.fit(X)
        X_test = scaler.transform(X_test)

        # 학습된 모델 로드 및 예측 수행
        model = keras.models.load_model('pima_model.keras')
        prediction = model.predict(X_test)
        res = prediction[0][0]
        res = np.round(res, 2)
        res = float(np.round(res * 100))

        return render_template('result.html', res=res)
    
    return render_template('prediction.html', form=form)

# 앱 실행
if __name__ == '__main__':
    app.run()
