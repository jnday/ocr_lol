import os
import cv2
from ultralytics import YOLO


# Step 1
#
# input: directory of images to process
# output: one .json file per image with bbox, class info, etc for each section detected


def predict_boxes():
    model_dir = 'models/'
    model_name = 'yolov8m-doclaynet.pt'
    model_path = model_dir + model_name
    input_dir = 'step_0/'
    output_dir = 'step_1/'
    os.makedirs(output_dir, exist_ok=True)
    valid_extensions = ('.jpg', '.png')

    model = YOLO(model_path)

    with os.scandir(input_dir) as iter:
        for entry in iter:
            if entry.is_file() and entry.name.endswith(valid_extensions):

                print(f"processing {entry.name}")
                
                img_name = entry.name.split('.')[0]
                json_path = output_dir + img_name + '.json'
                
                img = cv2.imread(entry.path, cv2.IMREAD_COLOR)
                #model = YOLO(model_path)
                result = model.predict(img)[0]

                with open(json_path, 'w') as json_file:
                    json_file.write(result.to_json())

                # lots of options with ultralytics, eg save image selection to jpg
                # but I can also do that elsewhere... so how to decide?
                if False:
                    img_path = output_dir + entry.name
                    # save annotated image if needed
                    result.save(img_path)
                    # or display it
                    result.show(img_path)

                    # stripped down version of csv save, might be useful for automation
                    txt_path = output_dir + img_name + '.txt'
                    result.save_txt(txt_path)
                    
                    # CSV dump for this page
                    csv_path = output_dir + img_name + '.csv'
                    with open(csv_path, 'w') as csv_file:
                        csv_file.write(result.to_csv())

def main():
    predict_boxes()

if __name__ == "__main__":
    main()

