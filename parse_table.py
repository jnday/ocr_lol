from ultralytics import YOLO
#from ultralytics import YOLOvv8
import xml.etree.ElementTree as ET

# https://huggingface.co/keremberke/yolov8m-table-extraction
# https://huggingface.co/keremberke/yolov8m-table-extraction/resolve/main/best.pt?download=true

# https://huggingface.co/foduucom/table-detection-and-extraction

def parse_table(image):
    # load model
    model = YOLO('https://huggingface.co/keremberke/yolov8m-table-extraction/resolve/main/best.pt?download=true')
    #model = YOLO.load()
    # set model parameters
    model.overrides['conf'] = 0.25  # NMS confidence threshold
    model.overrides['iou'] = 0.45  # NMS IoU threshold
    model.overrides['agnostic_nms'] = False  # NMS class-agnostic
    model.overrides['max_det'] = 1000  # maximum number of detections per image

    # perform inference
    results = model.predict(image)
    result=results[0]

    # observe results
    # print(result.boxes)
    # render = render_result(image, model, result)
    # render.show()

    return result.to_xml()




# def parse_foduu(image):
#     model = YOLOvv8.from_pretrained("foduucom/table-detection-and-extraction")
#     model.predict(source=image, save=True)