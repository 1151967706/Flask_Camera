from datetime import datetime
import random
import string

from flask import Blueprint, render_template, request, redirect, url_for,jsonify,session,flash

from exts import mail, db
from flask_mail import Message
from webmodels import  EmailCpatchaModel,UserModel
from .forms import RegisterForm,LoginForm
from werkzeug.security import generate_password_hash,check_password_hash


bp=Blueprint("user",__name__,url_prefix="/user")

@bp.route("/")
def index():
    return render_template("home.html")

@bp.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("user.login"))

@bp.route("/detect",methods=['GET','POST'])
def detect():
    if request.method=='GET':
        #判断用户是否登录
        print("1")
        user_id=session.get("user_id")
        print(user_id)
        if user_id:
            return render_template("detect.html")
        return render_template("login.html")
    else:
        return  0;


@bp.route("/login",methods=['GET','POST'])
def login():
    if request.method=='GET':
        return render_template("login.html")
    else:
        form=LoginForm(request.form)
        if form.validate():
            email=form.email.data
            password=form.password.data
            user=UserModel.query.filter_by(email=email).first()
            if user and check_password_hash(user.password,password):
                session['user_id']=user.id
                return redirect(url_for("user.detect"))
            else:
                flash("邮箱和密码不匹配")
                return redirect(url_for('user.login'))
        flash("邮箱或者密码格式错误")
        return redirect(url_for('user.login'))


@bp.route("/register",methods=['GET','POST'])
def register():
    if request.method=='GET':
        return render_template("register.html")
    else:
        form = RegisterForm(request.form)
        if form.validate():
            email = form.email.data
            username = form.username.data
            password = form.password.data
            hash_password=generate_password_hash(password)
            user = UserModel(email=email, username=username, password=hash_password)
            print(user)
            db.session.add(user)
            db.session.commit()
            print('1')
            return redirect(url_for("user.login"))
        else:
            print('2')
            return redirect(url_for("user.register"))



@bp.route("/captcha",methods=['POST'])
def my_mail():
    email=request.form.get("email")
    letters= string.ascii_letters+string.digits
    captcha = "".join(random.sample(letters, 4))
    if email:
        message = Message(
            subject="email test",
            recipients=[email],
            body=f"【djh问答】您的注册验证码是：{captcha}，请不要告诉任何人哦 "
        )
        mail.send(message)
        captcha_model=EmailCpatchaModel.query.filter_by(email=email).first()
        if captcha_model:
            captcha_model.captcha = captcha
            captcha_model.create_time=datetime.now()
            db.session.commit()
        else:
            captcha_model=EmailCpatchaModel(email=email,captcha=captcha)
            print(captcha_model.captcha)
            db.session.add(captcha_model)
            db.session.commit()
        return jsonify({"code":200})
    else:
        return jsonify({"code":400,"message":"请先传递邮箱"})

     [Jingyy]
     name-Jingyy
     baseurl = file:///mnt
     enabled-l
     gpgcheck-0
