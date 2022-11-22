# --------------------------------
# 卒業研究 2022 ベストショットレコメンドツール
# --------------------------------
from pickletools import floatnl
from re import S
from readline import append_history_file
from xxlimited import Xxo
import cv2
import csv
import torch
import math
import pprint
import numpy as np
import os
import dlib
from imutils import face_utils
from imutils import paths
import argparse
from asyncore import file_dispatcher
from distutils.file_util import move_file
from shutil import move
from unicodedata import name
from moviepy.editor import *

# print ('動画ファイル名を入力')
# name = input('>>')

# filename1 = name
# file_path = "/Users/arata/Documents/卒業研究/スクリプト/"+filename1+".mp4"

file_path = "/Users/arata/Documents/卒業研究/スクリプト/sample5.mp4"

# --------------------------------
# クラス群
# --------------------------------

def save_all_frames(video_path, dir_path, basename, ext='jpg'):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return

    os.makedirs(dir_path, exist_ok=True)
    base_path = os.path.join(dir_path, basename)

    digit = len(str(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))))

    n = 0
    
    print(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fn = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    fnl = len(str(int(fn)))

    print (fnl)
    while True:
        ret, frame = cap.read()
        
        #laplacian_thr =80

        #laplacian = cv2.Laplacian(frame, cv2.CV_64F)

        #print (laplacian)

        if ret: #and laplacian.var() >= laplacian_thr:
            cv2.imwrite('{}_{}.{}'.format(base_path, str(n).zfill(digit), ext), frame)
            n += 1
        else:
            return

def resFlameCount(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return

    else:
        a = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        fnl = len(str(int(a)))
        return fnl

def getFlameCount(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return

    return cap.get(cv2.CAP_PROP_FRAME_COUNT)

def getDistance(x1,y1,x2,y2):
    x3 = x2 - x1
    print ('x2-x1:'+str(x3))
    y3 = y2 - y1
    print ('y2-y1:'+str(y3))

    print (x1,y1,x2,y2)
    x = x3*x3
    print ('x^2:'+str(x))
    y = y3*y3
    print ('y^2:'+str(y))

    z = x + y
    print ('x+y:'+str(z))
    d = math.sqrt(z)
    print ('dis:'+ str(d))
    return d

def getAreaTriangle(x1,y1,x2,y2,x3,y3):

    a = getDistance(x1,y1,x2,y2)
    b = getDistance(x2,y2,x3,y3)
    c = getDistance(x1,y1,x3,y3)
    
    s1 = a+b+c
    s  = s1/2

    area1 = s-a
    area2 = s-b
    area3 = s-c
    area4 = s*area1*area2*area3
    area = math.sqrt(area4)
    print ('tri area:'+ str(area))
    return area

def getAreaSquare(x1,y1,x2,y2,x3,y3,x4,y4):
    
    xblist=[x1,x2,x3,x4]
    yblist=[y1,y2,y3,y4]

    xlist = sorted(xblist)
    ylist = sorted(yblist)
    print ('sort:'+str(xlist[0]),str(xlist[1]),str(xlist[2]),str(xlist[3]))
    print ('sort:'+str(ylist[0]),str(ylist[1]),str(ylist[2]),str(ylist[3]))
    x = int(xlist[3])-int(xlist[0])
    print ('dis:'+str(x))
    y = int(ylist[3])-int(ylist[0])
    print ('dis:'+str(y))
    area1 = x*y
    print (area1)
    area = area1/2
    print ('sq area:'+ str(area))
    return area

def getEyeArea(x1,y1,x2,y2,x3,y3,x4,y4,x5,y5,x6,y6):
    trl = getAreaTriangle(x1,y1,x2,y2,x6,y6)
    sq = getAreaSquare(x2,y2,x3,y3,x5,y5,x6,y6)
    trr = getAreaTriangle(x3,y3,x4,y4,x5,y5)
    print ('trl area:'+ str(trl))
    print ('sq area:'+ str(sq))
    print ('trr area:'+ str(trr))
    area = trl + sq + trr
    area = int(area)
    print ('eye area:'+ str(area))
    return area

def judgeLight(file):

    img=cv2.imread(file, cv2.IMREAD_GRAYSCALE) #1 グレイスケールで読み込み
    img=img.astype('float')
    img/=255

    sumGray=img.sum()  #2 全成分を足し合わせ
    meanGray=sumGray/(img.shape[0]*img.shape[1])  #3　規格化
    
    return meanGray
 
#無限(最大500回)

# --------------------------------
# 実験ファイル生成・時間指定・動画トリミング・全フレームカット
# --------------------------------


print ('パート番号を指定')
plot = input('>>')

save_folder = "/Users/arata/Documents/卒業研究/スクリプト/exp"+str(plot)
    # ディレクトリがない場合、作成する
if not os.path.exists(save_folder):
    print("ディレクトリを作成します")
    os.makedirs(save_folder)

if plot == 'end':
    print('次の処理に移ります......')
else:
    print ("開始時間を入力してください")
    x = input('>>')
    print ("終了時間を入力してください(終了の際には0を入力)")
    y = input('>>')
    print (x+":"+y)
    save_path1 = 'EditMovie'
    save_path2 = str(plot)
    save_path3 = '.mp4'
    save_path = save_folder+"/"+save_path1 + save_path2 +save_path3
    path =save_folder+"/flame_data/"
    print (save_path)
    clipdata = VideoFileClip(file_path).subclip(x,y)
    clipdata.write_videofile(save_path,fps=30)
    save_all_frames(save_path, path, 'sample_video_img', 'png')

st = x
et = y
save_path1 = 'EditMovie'
save_path2 = str(plot)
save_path3 = '.mp4'
save_path = save_folder+"/"+save_path1 + save_path2 +save_path3
print (type(save_path))

path = save_folder+"/flame_data/"
sub_dir="report/"
filesize = getFlameCount(save_path)
print (type(filesize))


# --------------------------------
# 全データリスト作成
# --------------------------------


data_list = [[0 for i in range(11)] for j in range(int(filesize)) ]


# --------------------------------
# 顔ランドマーク検出の前準備
# --------------------------------

face_detector = dlib.get_frontal_face_detector()
print ("cp1")
# 顔のランドマーク検出ツールの呼び出し
predictor_path = '/Users/arata/Documents/卒業研究/スクリプト/shape_predictor_68_face_landmarks.dat'
face_predictor = dlib.shape_predictor(predictor_path)
print ("cp2")

model = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True)
print(model.names)  # 検出できる物体の種類

for filenum in range(int(filesize)):
    fnl = resFlameCount(save_path)
    #filenumber = f'{filenum:0>3}'
    filenumber = str(filenum).zfill(fnl)
    image_file = "sample_video_img_"+filenumber+".png"
    image_path = path + image_file
    out_image_fpath = path + sub_dir
    out_image_path = path + sub_dir + image_file
    
    if not os.path.exists(out_image_fpath):
        print("ディレクトリを作成します")
        os.makedirs(out_image_fpath)

    data_list[filenum][0]= image_file

    print (image_path)
    print (out_image_path)

    image = cv2.imread(image_path) # 画像ファイル読み込み
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # グレースケースに変換

    laplacian = cv2.Laplacian(gray, cv2.CV_64F) #ラプラシアン値

    print("laplacian.var()の値は？",laplacian.var())
    data_list[filenum][1] = str(laplacian.var())

    if laplacian.var() < 130: # 閾値は100以下だとピンボケ画像と判定
        text = "Blurry"
    else:
        text = "Not Blurry"

    cv2.putText(image, "{}: {:.2f}".format(text, laplacian.var()), (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 3)

    cv2.imwrite(out_image_path, image)


  
    # 検出対象の画像の呼び込み
    img = cv2.imread(image_path)
    # 処理高速化のためグレースケール化(任意)
    img_gry = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print ("cp3")

# --------------------------------
# 1.類似度推定
# --------------------------------
    dos_list = [0]
    if(int(filenumber) > 0):
        print (filenumber)
        filenumber1 = str(filenum-1).zfill(fnl)
        image1 = cv2.imread(path+"sample_video_img_"+filenumber1+".png")
        image2 = cv2.imread(path+"sample_video_img_"+filenumber+".png")
        
        height = image1.shape[0]
        width = image1.shape[1]

        img_size = (int(width), int(height))

        # 比較するために、同じサイズにリサイズしておく
        image1 = cv2.resize(image1, img_size)
        image2 = cv2.resize(image2, img_size)
        
        image1_hist = cv2.calcHist([image1], [2], None, [256], [0, 256])
        image2_hist = cv2.calcHist([image2], [2], None, [256], [0, 256])
        gdos = np.count_nonzero(image1 == image2) / image2.size
        hdos = cv2.compareHist(image1_hist, image2_hist, 0)
        dos_list.append(gdos)
        dos_list.append(hdos)
        print("画素値比較の類似度：" + str(np.count_nonzero(image1 == image2) / image2.size))
        print("ヒストグラム比較の類似度：" + str(cv2.compareHist(image1_hist, image2_hist, 0)))



# --------------------------------
# 2.Yolov5 物体検出
# --------------------------------

    results = model(image_path)  # 画像パスを設定し、物体検出を行う
    objects = results.pandas().xyxy[0]  # 検出結果を取得
    save_path1 = path + "yolo_report/"
    results.save(save_dir=save_path1)

    # ディレクトリがない場合、作成する
    if not os.path.exists(save_path1):
        print("ディレクトリを作成します")
        os.makedirs(save_path1)

    obj_list=[]
    obj_infolist=[]
    hc = 0
    for i in range(len(objects)):
        name = objects.name[i]
        xmin = objects.xmin[i]
        ymin = objects.ymin[i]
        width = objects.xmax[i] - objects.xmin[i]
        height = objects.ymax[i] - objects.ymin[i]
        print(f"{i}, 種類:{name}, 座標x:{xmin}, 座標y:{ymin}, 幅:{width}, 高さ:{height}")
        obj_infolist.append(objects.name[i])
        if (name == 'person'):
            hc = hc + 1
        # obj_infolist.append(int(xmin))
        # obj_infolist.append(int(ymin))
        # obj_infolist.append(int(width))
        # obj_infolist.append(int(height))
    
   # obj_list.append(obj_infolist)

# --------------------------------
# 3.顔のランドマーク検出
# --------------------------------
# 顔検出
# ※2番めの引数はupsampleの回数。基本的に1回で十分。
    faces = face_detector(img_gry, 1)
    print ("cp4")
    hm = 0
    eyeareal=[]
    eyearear=[]
    facearea =[]
    eyearealeft = 0
    eyearearight = 0
    # 検出した全顔に対して処理
    for face in faces:
        eyex = []
        eyey = []
        facex = []
        facey = []
        # 顔のランドマーク検出
        landmark = face_predictor(img_gry, face)
        # 処理高速化のためランドマーク群をNumPy配列に変換(必須)
        landmark = face_utils.shape_to_np(landmark)
        print (str(hm+1)+"人目")
        hm=hm+1;
        # ランドマーク描画
        for (i, (x, y)) in enumerate(landmark):
            cv2.circle(img, (x, y), 1, (255, 0, 0), -1)
            #顔サイズ
            if (i == 0):
                facex.append(x)
                facey.append(y)
            elif(i == 7):
                facex.append(x)
                facey.append(y)
            elif(i == 9):
                facex.append(x)
                facey.append(y)
            elif(i == 16):
                facex.append(x)
                facey.append(y)

            #みぎめ36~41ひだりめ42~47
            if (i == 36):
                eyex.append(x)
                eyey.append(y)
            elif(i == 37):
                eyex.append(x)
                eyey.append(y)
            elif(i == 38):
                eyex.append(x)
                eyey.append(y)
            elif(i == 39):
                eyex.append(x)
                eyey.append(y)
            elif(i == 40):
                eyex.append(x)
                eyey.append(y)
            elif(i == 41):
                eyex.append(x)
                eyey.append(y)
            elif(i == 42):
                eyex.append(x)
                eyey.append(y)
            elif(i == 43):
                eyex.append(x)
                eyey.append(y)
            elif(i == 44):
                eyex.append(x)
                eyey.append(y)
            elif(i == 45):
                eyex.append(x)
                eyey.append(y)
            elif(i == 46):
                eyex.append(x)
                eyey.append(y)
            elif(i == 47):
                eyex.append(x)
                eyey.append(y)

            print (str(i+1)+"座標: x:"+str(x)+" y:"+str(y))
        
        if(hm >= 1):
            eyearealeft1 = getEyeArea(eyex[0],eyey[0],eyex[1],eyey[1],eyex[2],eyey[2],eyex[3],eyey[3],eyex[4],eyey[4],eyex[5],eyey[5])
            eyearearight1 = getEyeArea(eyex[6],eyey[6],eyex[7],eyey[7],eyex[8],eyey[8],eyex[9],eyey[9],eyex[10],eyey[10],eyex[11],eyey[11])
            facearea.append(str(getAreaSquare(facex[0],facey[0],facex[1],facey[1],facex[2],facey[2],facex[3],facey[3])))
            eyeareal.append(str(eyearealeft1))
            eyearear.append(str(eyearearight1))
            eyearealeft = (eyearealeft + eyearealeft1)/hm
            eyearearight = (eyearearight + eyearearight1)/hm
            print (str(i+1)+": 瞳面積: x:"+str(eyearealeft)+" y:"+str(eyearearight))
    data_list[filenum][2] = hc
    data_list[filenum][3] = obj_infolist
    data_list[filenum][4] = str(hm)
    data_list[filenum][5] = str(eyearealeft)
    data_list[filenum][6] = str(eyearearight)
    data_list[filenum][7] = facearea
    data_list[filenum][8] = eyeareal
    data_list[filenum][9] = eyearear
    data_list[filenum][10] = dos_list
 
# --------------------------------
# 3.結果表示
# --------------------------------
    print ("cp69")
    facefolder = save_folder+"/"+"facefolder"
    if not os.path.exists(facefolder):
        print("ディレクトリを作成します")
        os.makedirs(facefolder)
    cv2.imwrite(facefolder+'/face'+str(filenum)+'.png', img)
    print ("cp")
    

print('-----------------------------------')
a = 0

# --------------------------------
# 4.CSV結果出力
# --------------------------------

with open(save_folder+"/"+'datalist.csv', 'w') as csv_file:
    fieldnames = ['ファイル名','Laplacian値','人検知数','物体検知リスト','顔検知数','平均左目サイズ','平均右目サイズ','顔サイズ','左目サイズリスト','右目サイズリスト','類似度リスト']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()
    for num in data_list:       
        print("[ファイル名:"+ data_list[a][0] + ", Laplacian値:" + data_list[a][1]+ "，人検知数:" + str(data_list[a][2]) + "，物体検知リスト:" + str(data_list[a][3])+ "，顔検知数:" + data_list[a][4] + "，平均左目サイズ(人では右目):" + data_list[a][5] +"，平均右目サイズ(人では左目):" + data_list[a][6] +"，顔サイズ:" + str(data_list[a][7]) +"，左目サイズリスト:" + str(data_list[a][8])+"，右目サイズリスト:" + str(data_list[a][9])+"，類似度リスト:" + str(data_list[a][10])+"]")
        writer.writerow({'ファイル名': data_list[a][0],'Laplacian値': data_list[a][1],'人検知数':str(data_list[a][2]),'物体検知リスト':str(data_list[a][3]),'顔検知数':data_list[a][4],'平均左目サイズ':data_list[a][5],'平均右目サイズ':data_list[a][6],'顔サイズ':data_list[a][7],'左目サイズリスト':str(data_list[a][8]),'右目サイズリスト':str(data_list[a][9]),'類似度リスト':str(data_list[a][10])})
        a = a + 1
    writer.writerow({'ファイル名':file_path})
    writer.writerow({'ファイル名':str(st),'Laplacian値':str(et)})
   









