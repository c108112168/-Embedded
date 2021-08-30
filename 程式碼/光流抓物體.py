# -*- coding: utf-8 -*-
# OpenCV中的稠密光流 Farneback (動態)
import cv2
import numpy as np
import time

def my_sort():
    time.sleep(1.1234) # 用 sleep 模擬 my_sort() 運算時間

cap = cv2.VideoCapture('no_20210603_小港區_Part88.mp4')
step = 15
def draw_flow(gray, flow, step=3):
    h, w = gray.shape[:2]
    y, x = np.mgrid[step / 2:h:step, step / 2:w:step].reshape(2, -1).astype(int)
    fx, fy = flow[y, x].T
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    lines = np.vstack([x, y, x + fx, y + fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines)

    Drow_img = np.zeros_like(gray)
    cv2.line(gray, (100, 340), (250, 340), (0, 0, 255), 5)

    for (x1, y1), (x2, y2) in lines:

        if (mag[y1][x1] > 2) and (gray[y1][x1] > 150):  # 強度大於自訂範圍
            if (10 < (ang[y1][x1] * 180 / np.pi / 2)) and ((ang[y1][x1] * 180 / np.pi / 2) < 80):
                Drow_img[y1][x1] = 255

    return Drow_img

def draw_min_rect_circle(img, cnts):#繪製邊緣
    img = np.copy(img)
    for cnt in cnts:
        perimeter = cv2.arcLength(cnt, True)#計算周長
        x, y, w, h = cv2.boundingRect(cnt)
        if w>=50 :#去掉太大和太小的

            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), -1)  # 白色實心框


    return img

def cut_flow(img, cnts):  # 抓取光流圖片存檔
    img = np.copy(img)
    j = 0
    for cnt in cnts:

        x, y, w, h = cv2.boundingRect(cnt)
        perimeter = cv2.arcLength(cnt, True)  # 計算周長
        if perimeter >= 200:  # 去掉太大和太小的
            j = j + 1
            final_path = path + str(i) + "_" + str(j) + "_" + str(x) + "_" + str(y) + "_" + str(w) + "_" + str(h) + "_flow.bmp"
            crop_img = img[y:y + h, x:x + w]
            cv2.imwrite(str(final_path), crop_img)
    return img

def cut(img, cnts):  # 抓取原圖片存檔
    img = np.copy(img)
    j = 0
    for cnt in cnts:

        x, y, w, h = cv2.boundingRect(cnt)
        perimeter = cv2.arcLength(cnt, True)  # 計算周長

        if perimeter >= 200:  # 去掉太大和太小的
            j = j + 1
			#設定路徑名稱
            final_path = path + str(i) + "_" + str(j) + "_" + str(x) + "_" + str(y) + "_" + str(w) + "_" + str(h) + ".bmp"
			#剪取
            crop_img = img[y:y + h, x:x + w]
            cv2.imwrite(str(final_path), crop_img)

    return img

def cut_color(img, cnts):  # 抓取顏色圖片存檔
    img = np.copy(img)
    j = 0
    for cnt in cnts:

        x, y, w, h = cv2.boundingRect(cnt)
        perimeter = cv2.arcLength(cnt, True)  # 計算周長
        if perimeter >= 200:  # 去掉太大和太小的
            j = j + 1
            final_path = path + str(i) + "_" + str(j) + "_" + str(x) + "_" + str(y) + "_" + str(w) + "_" + str(h) + "_color.bmp"
            crop_img = img[y:y + h, x:x + w]
            cv2.imwrite(str(final_path), crop_img)
    return img




width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
FPS = int(cap.get(cv2.CAP_PROP_FPS))
print("Image Size: %d x %d , %d" % (width, height, FPS))

ret, im = cap.read()  # 第一幀
old_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)  # 灰階化

fourcc = 0x00000021  # cv2.VideoWriter_fourcc('H', '2', '6', '4')
videoWriter = cv2.VideoWriter('D:/output/Flow.mp4', fourcc, 30, (width, height))  # 建立 VideoWriter 物件，輸出影片至 Flow.mp4


#顏色設定
hsv = np.zeros_like(im)
hsv[..., 1] = 255

prevgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
#設定路徑
path = "D:/output/photo/"
i = 0 #計算路徑名稱所用到的數字(第幾幀)
while (cap.isOpened()):
    ret, im = cap.read()  # 第二幀
    i = i + 1
    if ret == True:
        t1 = time.time()
        my_sort()
        im_old = im  # 第一幀 = 第二幀 (原影片)
        new_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)  # 灰階化

        ret, thresh = cv2.threshold(new_gray.copy(), 160, 255, cv2.THRESH_BINARY)

        flow = cv2.calcOpticalFlowFarneback(old_gray, new_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)  # compute flow
        # print(flow.shape)

        old_gray = new_gray  # 第一幀 = 第二幀


        new_draw_flow = draw_flow(new_gray, flow)

        ##part222222222222222222222222222222222222222222222222

        ret, new_th = cv2.threshold(new_draw_flow, 127, 255, cv2.THRESH_BINARY)
        old_th = new_th  # 第一幀 = 第二幀 (光流二值)
        # 侵蝕膨脹
        kernel = np.ones((3, 3), np.uint8)
        dilation = cv2.dilate(old_th, kernel, iterations=10)
        erosion = cv2.erode(dilation, kernel, iterations=7)

        thresh = cv2.Canny(erosion, 50, 150)  # CANNY找邊緣
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        im_zero = np.zeros((height, width, 3), np.uint8)
        img = draw_min_rect_circle(im_zero, contours)



        #part333333333333333333333333333333333333

        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (11, 11), 0)
        flow = cv2.calcOpticalFlowFarneback(prevgray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        flow = cv2.GaussianBlur(flow, (11, 11), 0)  # 高斯濾波
        prevgray = gray

        # 設定箭頭繪製的資料
        h, w = gray.shape[:2]
        y, x = np.mgrid[step / 2:h:step, step / 2:w:step].reshape(2, -1).astype(int)
        fx, fy = flow[y, x].T
        lines = np.vstack([x, y, x + fx, y + fy]).T.reshape(-1, 2, 2)
        lines = np.int32(lines)
        line = []

        # 顏色
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        ret, new_th1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

        old_th1 = new_th1  # 第一幀 = 第二幀 (光流二值)

        for l in lines:
            dx = l[0][0] - l[1][0]  # l[0][0]為X點座標,l[1][0]為X點延伸的縣的另一點座標(dx為線的x位移量)
            dy = l[0][1] - l[1][1]
            line_X = l[0][0]
            line_Y = l[0][1]
            #print(old_th[line_Y][line_X])

            # 判斷是否為標記範圍內
            if ((old_th[line_Y][line_X]) == [255, 255, 255]).any():
                line.append(l)

        thresh = cv2.Canny(new_th1, 50, 150)  # CANNY找邊緣
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)





        # 剪取圖片(im2為原圖)

        cut(im_old, contours)
        cv2.polylines(im_old, line, 0, (0, 0, 255))
        cut_flow(im_old, contours)
        cut_color(bgr, contours)
        print('end' + str(i))
        t2 = time.time()
        print('time elapsed: ' + str(round(t2 - t1, 2)) + ' 秒')

        if cv2.waitKey(10) == 27:
            break
    else:
        break
print('end_all')
cap.release()
videoWriter.release()
cv2.destroyAllWindows()