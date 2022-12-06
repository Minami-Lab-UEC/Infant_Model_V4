from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, Table, MetaData
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.orm.exc import NoResultFound
import psycopg2
import re
from PIL import Image
import numpy as np
import pandas as pd

engine = create_engine('postgresql://taguchi:sdkogaken@localhost:5432/testsql')
# モデルの作成
# 説明のためファイル内に定義しますが、実際は別ファイル化して import します。

# まずベースモデルを生成します
Base = declarative_base()

data_size = 30

# 次にベースモデルを継承してモデルクラスを定義します

class New_Sub(Base):
    __tablename__ = 'sub_30'

    id = Column(Integer, primary_key=True)
    name = Column(String)
    deleted = Column(Boolean)

Base.metadata.create_all(engine)

# SQLAlchemy はセッションを介してクエリを実行する
Session = sessionmaker(bind=engine)
session = Session()

# 1レコードの追加
#session.add(Student(id=1, name='Suzuki', score=70))


for k in range(data_size):
    session.add(New_Sub(id=k))
session.commit()

for k in range(data_size):
    new_sub = session.query(New_Sub).filter(New_Sub.id==k).first()
    if(k == 0 or k == 1 or k == 2):
        new_sub.name = 'apple'
    if(k==3 or k == 4 or k == 5):
        new_sub.name = 'orange'
    if(k==6 or k == 7 or k == 8):
        new_sub.name = 'ball'
    if(k==9 or k == 10 or k == 11):
        new_sub.name = 'block'
    if(k==12 or k == 13 or k == 14):
        new_sub.name = 'ball'
    if(k==15 or k == 16 or k == 17):
        new_sub.name = 'dog'
    if(k==18 or k == 19 or k == 20):
        new_sub.name = 'cat'
    if(k==21 or k == 22 or k == 23):
        new_sub.name = 'bird'
    if(k==24 or k == 25 or k == 26):
        new_sub.name = 'tomato'
    if(k==27 or k == 28 or k == 29):
        new_sub.name = 'carrot'
    session.commit()
    new_sub.deleted = False
    session.commit()


"""
for k in range(15):
    s = './illust/' + str(k) + '.jpg'
    path = s

    rgb = rgb_ave(path)

    new_color = session.query(New_Color).filter(New_Color.id==k).first()
    new_color.r_top_left = rgb[0][0]
    new_color.r_top_center = rgb[1][0]
    new_color.r_top_right = rgb[2][0]
    new_color.r_middle_left = rgb[3][0]
    new_color.r_middle_center = rgb[4][0]
    new_color.r_middle_right = rgb[5][0]
    new_color.r_bottom_left = rgb[6][0]
    new_color.r_bottom_center = rgb[7][0]
    new_color.r_bottom_right = rgb[8][0]
    new_color.g_top_left = rgb[0][1]
    new_color.g_top_center = rgb[1][1]
    new_color.g_top_right = rgb[2][1]
    new_color.g_middle_left = rgb[3][1]
    new_color.g_middle_center = rgb[4][1]
    new_color.g_middle_right = rgb[5][1]
    new_color.g_bottom_left = rgb[6][1]
    new_color.g_bottom_center = rgb[7][1]
    new_color.g_bottom_right = rgb[8][1]
    new_color.b_top_left = rgb[0][2]
    new_color.b_top_center = rgb[1][2]
    new_color.b_top_right = rgb[2][2]
    new_color.b_middle_left = rgb[3][2]
    new_color.b_middle_center = rgb[4][2]
    new_color.b_middle_right = rgb[5][2]
    new_color.b_bottom_left = rgb[6][2]
    new_color.b_bottom_center = rgb[7][2]
    new_color.b_bottom_right = rgb[8][2]

# コミット（データ追加を実行）
session.commit()
"""


session.commit()
