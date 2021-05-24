import numpy as np
import cv2
import matplotlib.pyplot as plt
cap = cv2.VideoCapture('slow_2.avi')


# 角點設定參數
feature_params = dict(maxCorners=1,#數量(優先找最強的角點)
                      qualityLevel=0.1,#去掉(最好等級*數字)以下的角點
                      minDistance=10,#角之間的最小距離(不能太近)
                      blockSize=10)#

# lucas kanade(光流)設定參數
lk_params = dict(winSize=(30, 30),#每個位置的搜尋範圍大小
                 maxLevel=2,#最大金字塔等級??
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))#終止條件

# 隨機顏色產生
color = np.random.randint(0, 255, (200, 3))

# 找出角點
ret, old_frame = cap.read()#影片每幀讀取?
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)#灰階
p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

#做出一樣大小,但內容為0的複製(用來標記目標)
mask = np.zeros_like(old_frame)

while (1):#循環
    ret, frame = cap.read()#影片每幀讀取?
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)#灰階

    # calculate optical flow
    #p1找到的標記向量ㄝ, st是否找到相似的標記點, err錯誤?
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, #前一幀
                                           frame_gray, #當前
                                           p0, #要抓的標記
                                           None,
                                           **lk_params)

    # 設定good points(st=1時)
    good_new = p1[st == 1]
    good_old = p0[st == 1]
    p0 = p1[st == 1]
    # draw the tracks
    new_a = 0
    new_b = 0
    dy = [0] *1
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        #old_a = new_a
        old_b = new_b
        a, b = new.ravel()
        c, d = old.ravel()
        mask = cv2.line(mask, (a, b), (c, d), color[i].tolist(), 2)
        frame = cv2.circle(frame, (a, b), 5, color[i].tolist(), -1)
        #print (a, b)
        #new_a = a
        new_b = b
        #dx = old_a - new_a
        dy[i] = old_b - new_b
        print(dy[i])
        print(i)


    img = cv2.add(frame, mask)

    # 中位
    maxdy = np.mean(dy)
    print(maxdy)
    #exit()
    # 向量計算
    rows, cols, xx = img.shape
    # 平移矩阵M：[[1,0,x],[0,1,y]]
    M = np.float32([[1, 0, 0], [0, 1, maxdy]])
    dst = cv2.warpAffine(img, M, (cols, rows))
    #exit()

    cv2.imshow('frame', dst)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

    # Now update the previous frame and previous points
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1, 1, 2)







cap.release()
cv2.destroyAllWindows()