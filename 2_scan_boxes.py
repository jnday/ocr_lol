import os
import json
import cv2
from describe_img import process_blip, process_llama
from ocr_img import use_easyocr, use_tesseract
from parse_table import parse_table

# Step 2
#
# input: directory of .json files, one per image in previous step
# output: one .json file per input file, with the 'contents' value now set.

image_dir = 'step_0/'
data_dir = 'step_1/'
output_dir = 'step_2/'
os.makedirs(output_dir, exist_ok=True)
model_dir = 'models/'
valid_extensions = ('.json')


with os.scandir(data_dir) as iter:
    for entry in iter:
        if entry.is_file() and entry.name.endswith(valid_extensions):  # for each .json file in data_dir
            document_name = entry.name.removesuffix(valid_extensions)
            document_data = []   

            page_path = image_dir + document_name + '.jpg'
            page_image = cv2.imread(page_path, cv2.IMREAD_COLOR)

            with open(entry.path) as file:
                data = json.load(file)

            section_idx = 0
            for section in data:
                contents = []
                scores = []

                print(f"processing {document_name}:{section['name']}")

                # Section coords
                section_image = page_image[
                    int(section['box']['y1']) : int(section['box']['y2']), 
                    int(section['box']['x1']) : int(section['box']['x2'])
                ]
                
                # # save this step
                # section_path = f"{page_dir}{document_name}-{section_idx}.jpg"
                # ok = cv2.imwrite(section_path, section_image)

                match section['name']:
                    case 'Picture':
                        # send to multimodal
                        if True:
                            # Blip - fast but one word
                            section['blip-contents'] = process_blip(section_image)
                        else:
                            # Llamavision - amazingly slow but really good
                            section['llama-contents'] = process_llama(section_image)

                    case 'Table':
                        # send to table-specific
                        section['table-contents'] = parse_table(section_image)

                    case _: # treating everything else like 'Text'
                        # send to OCR
                        if True:
                            # EasyOCR
                            section['easyocr-contents'], section['easyocr-confidence'] = use_easyocr(section_image)
                        else:
                            # Tesseract
                            section['tesseract-contents'], section['tesseract-confidence'] = use_tesseract(section_image)
                
                document_data.append(section)
                section_idx += 1
                # section end
                   
            # document end
            filename = output_dir + document_name + '.json'
            with open(filename, 'w') as f:
                json.dump(document_data, f, indent=4)

