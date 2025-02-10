import os
from ultralytics import YOLO
import cv2

# Step 1
#
# input: directory of images to process
# output: one .json file per image with bbox, class info, etc for each section detected

# https://huggingface.co/hantian/yolo-doclaynet/resolve/main/yolov8m-doclaynet.pt
model_path = 'models/yolov8m-doclaynet.pt'
#input_dir = 'input/img/'
input_dir = 'step_0/'
output_dir = 'step_1/'
os.makedirs(output_dir, exist_ok=True)

valid_extensions = ('.jpg', '.png')
images = []

model = YOLO(model_path)

with os.scandir(input_dir) as iter:
    for entry in iter:
        if entry.is_file() and entry.name.endswith(valid_extensions):
            print(f"processing {entry.name}")
            img_save_path = output_dir + entry.name
            base_name = entry.name.removesuffix('.jpg')
            txt_save_path = output_dir + base_name + '.txt'
            json_save_path = output_dir + base_name + '.json'
            csv_save_path = output_dir + base_name + '.csv'
            img = cv2.imread(entry.path, cv2.IMREAD_COLOR)

            # https://docs.ultralytics.com/modes/predict/
            # https://docs.ultralytics.com/reference/engine/results/
            result = model.predict(img)[0]

            # # save annotated image if needed
            # result.save(img_save_path)

            # # stripped down version of csv save, might be useful for automation
            # result.save_txt(txt_save_path)
            
            # # spreadsheet dump for this page
            # with open(csv_save_path, 'w') as csv_file:
            #     csv_file.write(result.to_csv())

            # look up write vs dump
            with open(json_save_path, 'w') as json_file:
                json_file.write(result.to_json())

            # send to next stage as dataframe or similar
            #
            #