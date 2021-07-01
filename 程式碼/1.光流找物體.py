# -*- coding: utf-8 -*-
# OpenCV中的稠密光流 Farneback (動態)
import cv2
import numpy as np


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

    cv2.imshow("Drow_img", Drow_img)
    return Drow_img


# cap = cv2.VideoCapture(0)
# ret,im = cap.read()

cap = cv2.VideoCapture('MOV00271.mp4')

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
FPS = int(cap.get(cv2.CAP_PROP_FPS))
print("Image Size: %d x %d , %d" % (width, height, FPS))

ret, im = cap.read()  # 第一幀
old_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)  # 灰階化

fourcc = 0x00000021  # cv2.VideoWriter_fourcc('H', '2', '6', '4')
videoWriter = cv2.VideoWriter('D:/output/Flow_MOV00271.mp4', fourcc, 30, (width, height))  # 建立 VideoWriter 物件，輸出影片至 output.avi

while (cap.isOpened()):
    ret, im = cap.read()  # 第二幀

    if ret == True:


        new_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)  # 灰階化

        ret, thresh = cv2.threshold(new_gray.copy(), 160, 255, cv2.THRESH_BINARY)

        flow = cv2.calcOpticalFlowFarneback(old_gray, new_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)  # compute flow
        # print(flow.shape)

        old_gray = new_gray  # 第一幀 = 第二幀

        videoWriter.write(draw_flow(new_gray, flow))  # 輸出影片 要等...



        if cv2.waitKey(10) == 27:
            break
    else:
        break

cap.release()
videoWriter.release()
cv2.destroyAllWindows()