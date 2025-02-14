# OCR_lol

## Introduction

Generating an OCR Pipeline from a Hacker News comment.

I came across a comment on Hacker News that seemed like a straightforward guide to setting up an OCR pipeline, 
and **wanted to see if I could set that up by pasting the comment into a LLM**.

#### Difficulty:
   - Minimal Python experience
   - Everything runs locally
      - On my old laptop
         - i5-7, GTX 1050ti(4gb), 32bg ram
      - In WSL2
   - Each step generates an artifact

#### Results:
 
With PDF or JPG input this repo produces an XML document loosely conforming to [ALTO/PAGE schemas](https://kbnlresearch.github.io/xml-workshop/6_xmlformats_intro.html).
It does so by using several ML vision and language models in series to divide each page into labeled sections and then process each label category by content type.
It took about a week of evenings to hack together a "working" pipeline.  

This was a fun exercise. Python is enjoyable to work with, even though I spent a lot of time wrapping my head around Python's virtual environment situation.


## OP

The original post and comment chain that gave me the idea.

Post: https://news.ycombinator.com/item?id=42952605  
*Ingesting PDFs and why Gemini 2.0 changes everything*


Comment: https://news.ycombinator.com/item?id=42955414

<details>
<summary>Full text</summary>

```
This is using exactly the wrong tools at every stage of the OCR pipeline, and the cost is astronomical as a result.
You don't use multimodal models to extract a wall of text from an image. They hallucinate constantly the second you get past perfect 100% high-fidelity images.

You use an object detection model trained on documents to find the bounding boxes of each document section as _images_; each bounding box comes with a confidence score for free.

You then feed each box of text to a regular OCR model, also gives you a confidence score along with each prediction it makes.

You feed each image box into a multimodal model to describe what the image is about.

For tables, use a specialist model that does nothing but extract tables—models like GridFormer that aren't hyped to hell and back.

You then stitch everything together in an XML file because Markdown is for human consumption.

You now have everything extracted with flat XML markup for each category the object detection model knows about, along with multiple types of probability metadata for each bounding box, each letter, and each table cell.

You can now start feeding this data programmatically into an LLM to do _text_ processing, where you use the XML to control what parts of the document you send to the LLM.

You then get chunking with location data and confidence scores of every part of the document to put as meta data into the RAG store.

I've build a system that read 500k pages _per day_ using the above completely locally on a machine that cost $20k.
```
</details>



## Prompt

A few edits to the comment gives me a prompt I can paste into the machine.

<details>
<summary>Full text</summary>

```
Help me create an OCR pipeline from these directions:
 
 
You use an object detection model trained on documents to find the bounding boxes of each document section as images and each bounding box comes with a confidence score for free.
 
You then feed each box of text to a regular OCR model, also giving you a confidence score along with each prediction it makes.
 
You feed each image box into a multimodal model to describe what the image is about.
 
For tables, use a specialist model that does nothing but extract tables—models like GridFormer
 
You then stitch everything together in an XML file
 
You now have everything extracted with flat XML markup for each category the object detection model knows about, along with multiple types of probability metadata for each bounding box, each letter, and each table cell.
 
You can now start feeding this data programmatically into an LLM to do text processing, where you use the XML to control what parts of the document you send to the LLM.
 
You then get chunking with location data and confidence scores of every part of the document to put as meta data into the RAG store.
```

</details>

## Response

Following is the edited response from [gpt-4o](https://platform.openai.com/docs/models#gpt-4o), using 1174 tokens.

<div style="background-color:rgba(35, 10, 56, 0.05); border: 1px solid gray; border-radius: 5px; padding: 10px;">

```
Creating an OCR pipeline as described involves several sophisticated steps, integrating various machine learning models and data processing techniques. Below is a high-level outline of how you might implement such a pipeline using Python and relevant libraries:
```

### Step 0: Convert PDF to image(s)

*I added this step.*

<details>
<summary>
Expand step
</summary>

1. **Install Imagemagick and do This**: 
    ```sh
    convert -density 300 -channel RGB input/Weick2005.pdf input/weick.jpg
    ```

2. **Or Something Like This**:

    ```python
    from wand.image import Image
    with Image(filename='input/Test.pdf', resolution=300) as img:
        img.colorspace = 'rgb'
        img.save(filename='input/test.jpg')
    ```
</details>

### Step 1: Object Detection for Document Sections
<details>
<summary>
Expand step
</summary>

1. **Load Object Detection Model**: Use a pre-trained object detection model (e.g., YOLO, Faster R-CNN) suitable for detecting document sections.

    ```python
    model = YOLO('yolov8m-doclaynet.pt')
    ```

2. **Detect Document Sections**:
   - Input: Document image.
   - Output: Bounding boxes with confidence scores for each detected section.
   
   ```python
    img = cv2.imread('path/to/your_image.jpg', cv2.IMREAD_COLOR)
    result = model.predict(img)[0]
    print(result.boxes, result.scores)
   ```
</details>
   

### Step 2: OCR on Text Sections
<details>
<summary>
Expand step
</summary>

1. **Load OCR Model**: Use a robust OCR library like Tesseract or a deep learning-based OCR model (e.g., EasyOCR).

2. **Extract Text from Bounding Boxes**:
   - Loop through detected text sections and apply OCR.
   - Collect text with confidence scores for each section.

   ```python
   import pytesseract

   for box in text_boxes:
       x, y, w, h = box
       section_image = document_image[y:y+h, x:x+w]
       text_data = pytesseract.image_to_data(section_image, output_type=pytesseract.Output.DICT)
       section_text = text_data['text']
       ocr_confidences = text_data['conf']
   ```
</details>

### Step 3: Image Description with Multimodal Model
<details>
<summary>
Expand step
</summary>

1. **Load Multimodal Model**: Use a model like CLIP or another multimodal model to describe images.

2. **Describe Image Sections**:
   - Input: Image sections.
   - Output: Descriptions with confidence scores.

   ```python
   multimodal_model = load_multimodal_model()
   for box in image_boxes:
       x, y, w, h = box
       image_section = document_image[y:y+h, x:x+w]
       description, confidence = multimodal_model.describe(image_section)
   ```
</details>

### Step 4: Table Extraction
<details>
<summary>
Expand step
</summary>

1. **Load Table Extraction Model**: Use a specialized table extraction model like GridFormer.

2. **Extract Tables**:
   - Input: Document image or specific sections.
   - Output: Table data with confidence scores.

   ```python
   table_model = load_table_extraction_model()
   tables = table_model.extract(document_image)
   ```
</details>


### Step 5: Stitch Data into XML

<details>
<summary>
Expand step
</summary>

1. **Create XML Structure**: Organize extracted data into an XML format.

2. **Add Metadata**:
   - Include bounding box locations, confidence scores, and text data.

   ```python
   from lxml import etree

   root = etree.Element("Document")
   # Add sections with metadata
   for section in sections:
       section_elem = etree.SubElement(root, "Section")
       etree.SubElement(section_elem, "Text").text = section_text
       etree.SubElement(section_elem, "Confidence").text = str(ocr_confidences)

   # Add more elements for images, tables, etc.
   ```
</details>

### Step 6: Feed Data to an LLM
<details>
<summary>
Expand step
</summary>

1. **Prepare Data for LLM**: Use XML to select specific sections of the document for processing.

2. **Integrate with LLM**: Use an API or library like OpenAI's GPT to process text.

   ```python
   # Example pseudo-code for LLM interaction
   llm_response = llm_model.process(xml_data)
   ```
</details>

### Step 7: Store Metadata in RAG

#### NOTE: This wasn't part of the project.

<details>
<summary>
Expand step
</summary>

1. **Organize for Retrieval-Augmented Generation (RAG)**:
   - Structure data with metadata for efficient retrieval and processing.

2. **Store in RAG System**:
   - Metadata includes confidence scores, text, location information.

   ```python
   rag_store = RAGStore()
   rag_store.store(xml_with_metadata)
   ```
</details>

### Tools and Libraries:
- Object Detection: TensorFlow, PyTorch, Detectron2.
- OCR: Tesseract, EasyOCR.
- Multimodal Models: CLIP, Hugging Face Transformers.
- Table Extraction: GridFormer or similar.
- XML Processing: `lxml` or `xml.etree.ElementTree`.
- LLM Integration: OpenAI API, Transformers library.

```
This pipeline provides a comprehensive approach to document analysis, leveraging both traditional and advanced machine learning techniques to extract, organize, and utilize document data effectively.
```

</div>

# Notes

Money vs. Time

Control of input format

Quality of Image Descriptions
   - "logo" vs "wall of text"

Webcam in WSL via streaming mjpeg

Routing in Go or similar