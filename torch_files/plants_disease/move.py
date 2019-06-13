import json
import os
import shutil

from tqdm import tqdm

try:
    for i in range(0, 61):
        os.mkdir("../data/plant_disease_data/train/" + str(i))
        os.mkdir("../data/plant_disease_data/val/" + str(i))
except:
    pass

file_train = json.load(
    open("../data/plant_disease_data/temp/train/AgriculturalDisease_train_annotations.json", "r", encoding="utf-8"))
file_val = json.load(
    open("../data/plant_disease_data/temp/val/AgriculturalDisease_validation_annotations.json", "r", encoding="utf-8"))

for file in tqdm(file_train):
    filename = file["image_id"]
    origin_path = "../data/plant_disease_data/temp/train/images/" + filename
    ids = file["disease_class"]
    save_path = "../data/plant_disease_data/train/" + str(ids) + "/"
    shutil.copy(origin_path, save_path)

for file in tqdm(file_val):
    filename = file["image_id"]
    origin_path = "../data/plant_disease_data/temp/val/images/" + filename
    ids = file["disease_class"]
    save_path = "../data/plant_disease_data/val/" + str(ids) + "/"
    shutil.copy(origin_path, save_path)
