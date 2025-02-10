import os
import json
import easyocr
import cv2
import numpy as np
# import pytesseract as pt
# import time
from describe_img import process_blip, process_llama

# Step 2
#
# input: directory of .json files, one per image in previous step
# output: one .json file per input file, with the 'contents' value now set.


# https://huggingface.co/hantian/yolo-doclaynet/resolve/main/yolov8m-doclaynet.pt
# model_path = 'models/yolov8m-doclaynet.pt'

image_dir = 'step_0/'
data_dir = 'step_1/'
output_dir = 'step_2/'
os.makedirs(output_dir, exist_ok=True)
model_dir = 'models/'
valid_extensions = ('.json')


# OCR setup
reader = easyocr.Reader(['en'], model_storage_directory=model_dir)

with os.scandir(data_dir) as iter:
    for entry in iter:
        if entry.is_file() and entry.name.endswith(valid_extensions):
            # document start

            document_name = entry.name.removesuffix(valid_extensions)
            path = image_dir + document_name + '.jpg'
            page_image = cv2.imread(path, cv2.IMREAD_COLOR)
            document_data = []
            
            with open(entry.path) as file:
                data = json.load(file)

            for section in data:
                # bounding box start

                # Region Of Interest - bounting box of detected section
                # using the term here to differentiate clipping
                roi = page_image[
                    int(section['box']['y1']) : int(section['box']['y2']), 
                    int(section['box']['x1']) : int(section['box']['x2'])
                ]

                # init
                contents = []
                scores = []

                match section['name']:
                    case 'Picture':
                        print(section['name'], section['class'])
                        # send to multimodal
                        #process_blip(roi)
                        print("=============================================")
                        process_llama(roi)

                    case 'Table':
                        print(section['name'], section['class'])
                        # send to table-specific

                    case _: # treating everything else like 'Text'
                        # send to OCR

                        pass

                        # # pytesseract
                        # start_time = time.time()
                        # text_data = pt.image_to_data(roi, output_type=pt.Output.DICT)
                        # section['tesseract-contents'] = " ".join(text_data['text'])
                        # section['tesseract-confidence'] = np.mean(text_data['conf'], dtype='float64')
                        # end_time = time.time()
                        # section['tesseract-timer'] = end_time - start_time

                        # easyOCR
                        # result : [[int box coords], str ocr results, float confidence score ]
                        # start_time = time.time()
                        for result in reader.readtext(roi): 
                            contents.append(result[1])
                            scores.append(result[2])
                        section['easyocr-contents'] = "".join(contents)
                        section['easyocr-confidence'] = np.mean(scores, dtype='float64')
                        # end_time = time.time()
                        # section['easyocr-timer'] = end_time - start_time


                document_data.append(section)    


            filename = output_dir + document_name + '.json'
            with open(filename, 'w') as f:
                json.dump(document_data, f, indent=4)


