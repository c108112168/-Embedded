import numpy as np
import cv2
import os
import shutil

def predict(deepModel, feature):
    label = str(int(deepModel["model"].predict(feature)[1][0][0]))
    #print(label)
    try:
        return predict(deepModel[label], feature)
    except KeyError:
        return label


# ------------------------ Define ------------------------#
print("============================================== Start .")
# ------------------------ 這個路徑的資料夾image_data裡面放入和分類一樣數量的資料夾(空的喔,名字要和訓練部份一樣,但是要是下面部分的命名不一樣的話記得要改[看這個變數predict_y])以及一個放未訓練圖片的資料夾
iFilePath = "./image_data/"

path_list = [iFilePath + filename + '/' for filename in os.listdir(iFilePath)]

hogParams = {
    '_winSize': (64, 64),
    '_blockSize': (16, 16),
    '_blockStride': (8, 8),
    '_cellSize': (8, 8),
    '_nbins': 9
}
HoG = cv2.HOGDescriptor(**hogParams)

print("=============================================== Model input . ")

deep_model = {}

# ------------------------ 這個是你的模型的路徑
model_path = "./2021_08_16_18_11_04-useH-notUseR/"
# ------------------------

model_name_list = os.listdir(model_path)
model_name_list.sort(key=lambda m: len(m.split("-")))
for model_name in model_name_list:
    model_split = model_name[:-4].split("-")[1:]
    if len(model_split) == 0:
        deep_model = {
            "model": cv2.ml.SVM_load(model_path + model_name)
        }
    else:
        root = deep_model
        for category in model_split:
            try:
                root = root[category]
            except KeyError:
                root[category] = {}
        root[category] = {
            "model": cv2.ml.SVM_load(model_path + model_name)
        }

# ------------------------ start ------------------------#
print("Image reading ...")
total_image_path_list = []

for path in path_list:
    image_path_list = [path + filename for filename in os.listdir(path)]
    total_image_path_list = total_image_path_list + image_path_list

len_image_path_list = len(total_image_path_list)


same_number = 0
diff_number = 0

predict_result = []
actually_result = []
i = 0
for path in total_image_path_list:
    if("desktop" not in path):
        i=i+1
        apple = "test" + str(i)
        #測試的
        #print(apple)
        #print(path)
        image = cv2.imread(path, 0)

        resize_image = cv2.resize(image, (64, 64))

        HOG_data = HoG.compute(resize_image).ravel()
        HOG_data = np.array([HOG_data], dtype=np.float32)
        predict_y = predict(deep_model, HOG_data)

        path_cut_1, path_cut_2, path_cut_3, path_cut_4 = path.split("/")

        print(" %s is from %s , predict is %s " % (path, path_cut_3, predict_y))

        #移動檔案
        new_path = "./image_data/" + str(predict_y) + "/"
        shutil.move(path, new_path)

        predict_result.append(predict_y)
        actually_result.append(path_cut_3)





from sklearn.metrics import confusion_matrix


category_array = os.listdir(iFilePath)
category_array.sort()
res = confusion_matrix(actually_result, predict_result, labels=category_array)
print(category_array)