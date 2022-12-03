from exts import  db
from datetime import datetime

class EmailCpatchaModel(db.Model):
    __tablename__ = "email_captcha"
    id=db.Column(db.Integer,primary_key=True,autoincrement=True)
    email=db.Column(db.String(100),nullable=False,unique=True)
    captcha=db.Column(db.String(10),nullable=False)
    create_time=db.Column(db.DateTime,default=datetime.now)

class UserModel(db.Model):
    __tablename__ = "user"
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    username=db.Column(db.String(100),nullable=False,unique=True)
    email = db.Column(db.String(100), nullable=False, unique=True)
    password=db.Column(db.String(200),nullable=False)  #密码需要大于200，否则加密后的长度会被减少报错
    join_time=db.Column(db.DateTime,default=datetime.now)