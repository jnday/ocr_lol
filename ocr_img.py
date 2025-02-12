import pytesseract
import easyocr
import numpy as np

# pytesseract
def use_tesseract(image):
    text_data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
    return " ".join(text_data['text']), np.mean(text_data['conf'], dtype='float64')

# easyOCR
# result : [[int box coords], str ocr results, float confidence score ]
def use_easyocr(image):
    model_dir = 'models/' # todo: classes and self
    reader = easyocr.Reader(['en'], model_storage_directory=model_dir)
    contents=[]
    scores=[]
    for result in reader.readtext(image): 
        contents.append(result[1])
        scores.append(result[2])
    return  "".join(contents), np.mean(scores, dtype='float64')

