from sqlalchemy import create_engine, Column, Integer, String, Float, Table, MetaData
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

class New_Taste(Base):
    __tablename__ = 'new_taste_30'

    id = Column(Integer, primary_key=True)
    umami_glutamic_acid = Column(Float)
    salty_salt = Column(Float)
    acidity_organic_acid = Column(Float)
    acidity_aspartic_acid = Column(Float)
    bitter_caffeine = Column(Float)
    bitter_phenylalanine = Column(Float)
    bitter_tyrosine = Column(Float)
    bitter_arginine = Column(Float)
    bitter_isoleucine = Column(Float)
    bitter_leucine = Column(Float)
    bitter_valine = Column(Float)
    bitter_methionine = Column(Float)
    bitter_histidine = Column(Float)
    sweet_fructose = Column(Float)
    sweet_glucose = Column(Float)
    sweet_galactose = Column(Float)
    sweet_sucrose = Column(Float)
    sweet_maltose = Column(Float)
    sweet_lactose = Column(Float)
    sweet_trehalose = Column(Float)
    sweet_sorbitol = Column(Float)
    sweet_mannitol = Column(Float)
    sweet_glycine = Column(Float)
    sweet_alanine = Column(Float)
    sweet_threonine = Column(Float)
    sweet_proline = Column(Float)
    sweet_serine = Column(Float)

class Sub_Taste():
    
    def __init__(self,obj_name):
        if obj_name == False:
            self.obj_umami = [-1] * 2
            self.obj_salty = [-1] * 2
            self.obj_acidity = [-1] * 3
            self.obj_bitter = [-1] * 10
            self.obj_sweet = [-1] * 15
            return

        # CSV
        japanese_food  = pd.read_csv('../food/japanese_food.csv')
        sugar = pd.read_csv('../food/sugar.csv')
        amino = pd.read_csv('../food/amino.csv')

        object_name = obj_name

        umami = amino[['食品名','グルタミン酸']]
        object_umami = umami[umami['食品名'].str.contains(object_name, na=False)]
        object_umami = object_umami.reset_index(drop=True).head(1)
        #確認用
        #print(object_umami)

        salty = japanese_food[['食品名','食塩相当量']]
        object_salty = salty[salty['食品名'].str.contains(object_name, na=False)]
        object_salty = object_salty.reset_index(drop=True).head(1)
        #確認用
        #print(object_salty)

        acidity_1 = japanese_food[['食品名','有機酸']]
        acidity_2 = amino[['食品名','アスパラギン酸']]
        object_acidity_1 = acidity_1[acidity_1['食品名'].str.contains(object_name, na=False)]
        object_acidity_2 = acidity_2[acidity_2['食品名'].str.contains(object_name, na=False)]
        object_acidity_1 = object_acidity_1.reset_index(drop=True).head(1)
        object_acidity_2 = object_acidity_2.reset_index(drop=True).head(1)
        #確認用
        #print(object_acidity_1)
        #print(object_acidity_2)

        object_acidity = pd.concat([object_acidity_1.reset_index(drop=True).head(1), object_acidity_2.reset_index(drop=True).head(1).drop("食品名", axis=1)], axis=1)
        #確認用
        #print(object_acidity)

        bitter_1 = japanese_food[['食品名','カフェイン']]
        bitter_2 = amino[['食品名','フェニルアラニン','チロシン','アルギニン','イソロイシン','ロイシン','バリン','メチオニン','ヒスチジン']]
        object_bitter_1 = bitter_1[bitter_1['食品名'].str.contains(object_name, na=False)]
        object_bitter_2 = bitter_2[bitter_2['食品名'].str.contains(object_name, na=False)]
        object_bitter_1 = object_bitter_1.reset_index(drop=True).head(1)
        object_bitter_2 = object_bitter_2.reset_index(drop=True).head(1)
        #確認用
        #print(object_bitter_1)
        #print(object_bitter_2)

        object_bitter = pd.concat([object_bitter_1.reset_index(drop=True).head(1), object_bitter_2.reset_index(drop=True).head(1).drop("食品名", axis=1)], axis=1)
        #確認用
        #print(object_bitter)

        sweet_1 = sugar[['食品名','果糖','ぶどう糖','ガラクトース','しょ糖','麦芽糖','乳糖','トレハロース','ソルビトール','マンニトール']]
        sweet_2 = amino[['食品名','グリシン','アラニン','トレオニン（スレオニン）','プロリン','セリン']]
        object_sweet_1 = sweet_1[sweet_1['食品名'].str.contains(object_name, na=False)]
        object_sweet_2 = sweet_2[sweet_2['食品名'].str.contains(object_name, na=False)]
        object_sweet_1 = object_sweet_1.reset_index(drop=True).head(1)
        object_sweet_2 = object_sweet_2.reset_index(drop=True).head(1)
        #確認用
        #print(object_sweet_1)
        #print(object_sweet_2)

        object_sweet = pd.concat([object_sweet_1.reset_index(drop=True).head(1), object_sweet_2.reset_index(drop=True).head(1).drop("食品名", axis=1)], axis=1)
        #確認用
        #print(object_sweet)

        #データをfloat型に変換
        for i,name in enumerate(object_umami.columns):
            data = object_umami.loc[0, name]
            if ('-' in data):
                object_umami.at[0, name] = '-1'
            elif ('(' in data):
                data = data.replace('(',"").replace(')',"")
                object_umami.at[0, name] = data
            elif ('Tr' in data):
                object_umami.at[0, name] = '0.01'

        for i,name in enumerate(object_umami.columns):
            if(i != 0):
                object_umami[name] = object_umami[name].astype(float)
                data = object_umami.loc[0, name]
                object_umami.at[0, name] = float(object_umami.at[0, name])
        #確認用
        #print(object_umami)

        for i,name in enumerate(object_salty.columns):
            data = object_salty.loc[0, name]
            if ('-' in data):
                object_salty.at[0, name] = '-1'
            elif ('(' in data):
                data = data.replace('(',"").replace(')',"")
                object_salty.at[0, name] = data
            elif ('Tr' in data):
                object_salty.at[0, name] = '0.01'

        for i,name in enumerate(object_salty.columns):
            if(i != 0):
                object_salty[name] = object_salty[name].astype(float)
                data = object_salty.loc[0, name]
                object_salty.at[0, name] = float(object_salty.at[0, name])
        #確認用
        #print(object_salty)

        for i,name in enumerate(object_acidity.columns):
            data = object_acidity.loc[0, name]
            if ('-' in data):
                object_acidity.at[0, name] = '-1'
            elif ('(' in data):
                data = data.replace('(',"").replace(')',"")
                object_acidity.at[0, name] = data
            elif ('Tr' in data):
                object_acidity.at[0, name] = '0.01'

        for i,name in enumerate(object_acidity.columns):
            if(i != 0):
                object_acidity[name] = object_acidity[name].astype(float)
                data = object_acidity.loc[0, name]
                object_acidity.at[0, name] = float(object_acidity.at[0, name])
        #確認用
        #print(object_acidity)

        for i,name in enumerate(object_bitter.columns):
            data = object_bitter.loc[0, name]
            if ('-' in data):
                object_bitter.at[0, name] = '-1'
            elif ('(' in data):
                data = data.replace('(',"").replace(')',"")
                object_bitter.at[0, name] = data
            elif ('Tr' in data):
                object_bitter.at[0, name] = '0.01'

        for i,name in enumerate(object_bitter.columns):
            if(i != 0):
                object_bitter[name] = object_bitter[name].astype(float)
                data = object_bitter.loc[0, name]
                object_bitter.at[0, name] = float(object_bitter.at[0, name])
        #確認用
        #print(object_bitter)

        for i,name in enumerate(object_sweet.columns):
            data = object_sweet.loc[0, name]
            if ('-' in data):
                object_sweet.at[0, name] = '-1'
            elif ('(' in data):
                data = data.replace('(',"").replace(')',"")
                object_sweet.at[0, name] = data
            elif ('Tr' in data):
                object_sweet.at[0, name] = '0.01'

        for i,name in enumerate(object_sweet.columns):
            if(i != 0):
                object_sweet[name] = object_sweet[name].astype(float)
                data = object_sweet.loc[0, name]
                object_sweet.at[0, name] = float(object_sweet.at[0, name])
        #確認用
        #print(object_sweet)

        self.obj_umami = object_umami.values.tolist()[0]
        self.obj_salty = object_salty.values.tolist()[0]
        self.obj_acidity = object_acidity.values.tolist()[0]
        self.obj_bitter = object_bitter.values.tolist()[0]
        self.obj_sweet = object_sweet.values.tolist()[0]
        #print(self.obj_umami)

        #ノイズ用
        """
        np.random.seed(0)

        random = np.random.uniform(-10, 10, 3)
        print(random)

        noise = random * 0.01
        print(noise)
        """

Base.metadata.create_all(engine)

# SQLAlchemy はセッションを介してクエリを実行する
Session = sessionmaker(bind=engine)
session = Session()

# 1レコードの追加
#session.add(Student(id=1, name='Suzuki', score=70))


for k in range(data_size):
    session.add(New_Taste(id=k))
session.commit()


sub_taste = Sub_Taste(obj_name='りんご')

for k in range(data_size):
    if (k == 0 or k == 1 or k == 2):
       sub_taste = Sub_Taste(obj_name='りんご')
    elif (k == 3 or k == 4 or k == 5):
       sub_taste = Sub_Taste(obj_name='みかん')
    elif (k == 24 or k == 25 or k == 26):
       sub_taste = Sub_Taste(obj_name='トマト')
    elif (k == 27 or k == 28 or k == 29):
       sub_taste = Sub_Taste(obj_name='にんじん')
    else:
       sub_taste = Sub_Taste(obj_name=False)

    new_taste = session.query(New_Taste).filter(New_Taste.id==k).first()
    new_taste.umami_glutamic_acid = sub_taste.obj_umami[1]
    new_taste.salty_salt = sub_taste.obj_salty[1]
    new_taste.acidity_organic_acid = sub_taste.obj_acidity[1]
    new_taste.acidity_aspartic_acid = sub_taste.obj_acidity[2]
    new_taste.bitter_caffeine = sub_taste.obj_bitter[1]
    new_taste.bitter_phenylalanine = sub_taste.obj_bitter[2]
    new_taste.bitter_tyrosine = sub_taste.obj_bitter[3]
    new_taste.bitter_arginine = sub_taste.obj_bitter[4]
    new_taste.bitter_isoleucine = sub_taste.obj_bitter[5]
    new_taste.bitter_leucine = sub_taste.obj_bitter[6]
    new_taste.bitter_valine = sub_taste.obj_bitter[7]
    new_taste.bitter_methionine = sub_taste.obj_bitter[8]
    new_taste.bitter_histidine = sub_taste.obj_bitter[9]
    new_taste.sweet_fructose = sub_taste.obj_sweet[1]
    new_taste.sweet_glucose = sub_taste.obj_sweet[2]
    new_taste.sweet_galactose = sub_taste.obj_sweet[3]
    new_taste.sweet_sucrose = sub_taste.obj_sweet[4]
    new_taste.sweet_maltose = sub_taste.obj_sweet[5]
    new_taste.sweet_lactose = sub_taste.obj_sweet[6]
    new_taste.sweet_trehalose = sub_taste.obj_sweet[7]
    new_taste.sweet_sorbitol = sub_taste.obj_sweet[8]
    new_taste.sweet_mannitol = sub_taste.obj_sweet[9]
    new_taste.sweet_glycine = sub_taste.obj_sweet[10]
    new_taste.sweet_alanine = sub_taste.obj_sweet[11]
    new_taste.sweet_threonine = sub_taste.obj_sweet[12]
    new_taste.sweet_proline = sub_taste.obj_sweet[13]
    new_taste.sweet_serine = sub_taste.obj_sweet[14]

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
