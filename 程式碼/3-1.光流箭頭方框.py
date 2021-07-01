import cv2
import numpy as np

step = 15
def draw_min_rect_circle(img, cnts):  # 繪製邊緣
    img = np.copy(img)
    for cnt in cnts:
        perimeter = cv2.arcLength(cnt, True)  # 計算周長
        if perimeter >= 200:  # 去掉太大和太小的
            cv2.drawContours(img, [cnt, ], -1, (255, 0, 0), 2)  # blue
            hull = cv2.convexHull(cnt)
            # cv2.polylines(img, [hull, ], True, (0, 0, 255), 2)  # red

    return img

def cut(img, cnts):  # 抓取圖片存檔
    img = np.copy(img)
    j = 0
    for cnt in cnts:

        x, y, w, h = cv2.boundingRect(cnt)
        perimeter = cv2.arcLength(cnt, True)  # 計算周長
        if perimeter >= 200:  # 去掉太大和太小的
            j = j + 1
            final_path = path + str(i) + "_" + str(j) + ".jpg"  # 設定路徑名稱
            crop_img = img[y:y + h, x:x + w] # 割取圖片
            cv2.imwrite(str(final_path), crop_img)
    return img


cap = cv2.VideoCapture('check_MOV00271.mp4')  # 光流輸出影片(有物件的部分)
cap2 = cv2.VideoCapture('MOV00271.mp4')  # 原影片
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
FPS = int(cap.get(cv2.CAP_PROP_FPS))
print("Image Size: %d x %d , %d" % (width, height, FPS))

ret, im = cap.read()  # 光流第一幀
ret, im2 = cap2.read()  # 原影片 第一幀
prevgray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

ret, old_th = cv2.threshold(im, 127, 255, cv2.THRESH_BINARY)  # 第一幀二值化

fourcc = 0x00000021  # cv2.VideoWriter_fourcc('H', '2', '6', '4')
videoWriter = cv2.VideoWriter('D:/output/final_Gaussian.mp4', fourcc, 30,
                              (width, height))  # 建立 VideoWriter 物件，輸出影片至 output.avi
path = "D:/output/photo/object"
i = 0
while (cap.isOpened()):
    i = i + 1
    ret, im = cap.read()  # 光流第二幀
    ret, im2_new = cap2.read()  # 原第二幀

    if ret == True:
        gray = cv2.cvtColor(im2_new, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (11, 11), 0)
        flow = cv2.calcOpticalFlowFarneback(prevgray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        flow = cv2.GaussianBlur(flow, (11, 11), 0)
        prevgray = gray

        #繪製
        h, w = gray.shape[:2]
        y, x = np.mgrid[step / 2:h:step, step / 2:w:step].reshape(2, -1).astype(int)
        fx, fy = flow[y, x].T
        lines = np.vstack([x, y, x + fx, y + fy]).T.reshape(-1, 2, 2)
        lines = np.int32(lines)

        line = []




        ret, new_th = cv2.threshold(im, 127, 255, cv2.THRESH_BINARY)
        im2 = im2_new  # 第一幀 = 第二幀 (原影片)
        old_th = new_th  # 第一幀 = 第二幀 (光流二值)

        # 侵蝕膨脹調整
        kernel = np.ones((3, 3), np.uint8)
        erosion = cv2.erode(old_th, kernel, iterations=0)
        #dilation = cv2.dilate(erosion, kernel, iterations=15)
        #erosion = cv2.erode(dilation, kernel, iterations=15)
        dilation = cv2.dilate(erosion, kernel, iterations=0)

        for l in lines:
            dx = l[0][0] - l[1][0]#l[0][0]為X點座標,l[1][0]為X點延伸的縣的另一點座標(dx為線的x位移量)
            dy = l[0][1] - l[1][1]
            line_X = l[0][0]
            line_Y = l[0][1]
            print(dilation[line_Y][line_X])

            #判斷是否為標記範圍內
            if ((dilation[line_Y][line_X]) == [255, 255, 255]).any():
                line.append(l)

        thresh = cv2.Canny(new_th, 50, 150)  # CANNY找邊緣
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        img = draw_min_rect_circle(im2, contours)
        cv2.polylines(img, line, 0, (0, 0, 255))
        cv2.imshow('erosion', erosion)  # 輸出經過膨脹侵蝕之二值化圖
        cv2.imshow('img', img)  # 輸出繪製後
        cut(img, contours)

        videoWriter.write(img)  # 輸出影片 要等...

        # cv2.imshow('Optical flow',draw_flow(new_gray,flow))

        if cv2.waitKey(10) & 0xFF == ord('q'):  # 按Q暫停
            break
    else:
        break
    if (cv2.waitKey(30) >= 0):
        cv2.waitKey(0)
cap.release()
videoWriter.release()
cv2.destroyAllWindows()