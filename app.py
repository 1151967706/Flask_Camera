from importlib import import_module
import config
from blueprints import qa_bp,user_bp
from flask import Flask, render_template, Response,session, g
import os
from exts import  db,mail
from flask_migrate import Migrate
from webmodels import UserModel


if os.environ.get('CAMERA'):
    Camera=import_module('camera_'+os.environ['CAMERA']).Camera
else:
    from camera import Camera

app = Flask(__name__)
app.config.from_object(config)

db.init_app(app)
mail.init_app(app)
migrate=Migrate(app,db)

app.register_blueprint(qa_bp)
app.register_blueprint(user_bp)


@app.before_request
def before_request():
    user_id=session.get("user_id")
    if user_id:
        try:
            user=UserModel.query.get(user_id)
            g.user=user
        except:
            g.user=None

@app.context_processor
def context_processor():
    if hasattr(g,"user"):
        return {"user":g.user}
    else:
        return {}

@app.route('/detect')
def detect_mi():
    #获取用户登录状态
    user_id = session.get("user_id")
    print(user_id)
    #如果用户为登录状态，进入检测页面，else进入登录页面
    if user_id:
        return render_template("detect.html")
    return render_template("login.html")
    #return render_template('detect.html')

@app.route('/')
def hello_world():  # put application's code here
    # return render_template('detect.html')   #用于渲染页面
    return render_template('home.html')   #用于渲染页面

def gen(camera):
    """"视频流生成"""
    while True:
        frame=camera.get_frame()
        yield (b'--frame\r\n'
               b'Content=Type:image/jpeg\r\n\r\n'+frame+b'\r\n')  #设置请求媒体信息

@app.route('/video_feed')
def video_feed():
    """视频流的路线，将其放在img标记的src属性中"""
    return Response(gen(Camera()),mimetype='multipart/x-mixed-replace;boundary=frame')



if __name__ == '__main__':
    app.run()

    app.run(host='0.0.0.0',port=5001,threaded=True) #threaded=True为开启多线程
