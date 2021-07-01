import cv2
import numpy as np

def draw_min_rect_circle(img, cnts):#繪製邊緣
    img = np.copy(img)
    for cnt in cnts:
        perimeter = cv2.arcLength(cnt, True)#計算周長
        x, y, w, h = cv2.boundingRect(cnt)
        if w>=50 :#去掉太大和太小的

            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), -1)  # 白色實心框


    return img


cap = cv2.VideoCapture('Flow_MOV00271.mp4')#光流輸出影片
cap2 = cv2.VideoCapture('MOV00271.mp4')#原影片
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
FPS = int(cap.get(cv2.CAP_PROP_FPS))
print("Image Size: %d x %d , %d" % (width, height, FPS))

ret, im = cap.read()  # 光流第一幀
ret, im2 = cap2.read()  # 原影片 第一幀
ret, old_th = cv2.threshold(im, 127, 255, cv2.THRESH_BINARY)#第一幀二值化

fourcc = 0x00000021  # cv2.VideoWriter_fourcc('H', '2', '6', '4')
videoWriter = cv2.VideoWriter('D:/output/check_MOV00271.mp4', fourcc, 30,
                              (width, height))  # 建立 VideoWriter 物件，輸出影片至 output.avi

while (cap.isOpened()):
    ret, im = cap.read()  # 光流第二幀
    ret, im2_new = cap2.read()  # 原第二幀
    if ret == True:

        cv2.imshow('im', im2_new)

        ret, new_th = cv2.threshold(im, 127, 255, cv2.THRESH_BINARY)
        im2 = im2_new  # 第一幀 = 第二幀 (原影片)
        old_th = new_th  # 第一幀 = 第二幀 (光流二值)

        #侵蝕膨脹
        kernel = np.ones((3, 3), np.uint8)
        dilation = cv2.dilate(old_th, kernel, iterations=10)
        erosion = cv2.erode(dilation, kernel, iterations=7)


        thresh = cv2.Canny(erosion, 50, 150) #CANNY找邊緣
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        im_zero = np.zeros((height, width, 3), np.uint8)
        img = draw_min_rect_circle(im_zero, contours)

        cv2.imshow('erosion', erosion)#輸出經過膨脹侵蝕之二值化
        cv2.imshow('img', img)#輸出繪製後

        videoWriter.write(img)  # 輸出影片 要等...

        # cv2.imshow('Optical flow',draw_flow(new_gray,flow))

        if cv2.waitKey(10) & 0xFF == ord('q'):#按Q暫停
            break
    else:
        break
    if (cv2.waitKey(30) >= 0):
        cv2.waitKey(0)
cap.release()
videoWriter.release()
cv2.destroyAllWindows()