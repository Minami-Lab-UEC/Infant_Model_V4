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

data_size = 15


# 次にベースモデルを継承してモデルクラスを定義します

class New_Size(Base):
    __tablename__ = 'new_size'

    id = Column(Integer, primary_key=True)
    size = Column(Integer)

class New_Hardness(Base):
    __tablename__ = 'new_hardness'

    id = Column(Integer, primary_key=True)
    hardness = Column(Integer)

Base.metadata.create_all(engine)

# SQLAlchemy はセッションを介してクエリを実行する
Session = sessionmaker(bind=engine)
session = Session()

# 1レコードの追加
#session.add(Student(id=1, name='Suzuki', score=70))


for k in range(data_size):
    session.add(New_Size(id=k))
    session.add(New_Hardness(id=k))
session.commit()

for k in range(data_size):
    new_size = session.query(New_Size).filter(New_Size.id==k).first()
    new_size.size = -1
    new_hardness = session.query(New_Hardness).filter(New_Hardness.id==k).first()
    new_hardness.hardness = -1


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
