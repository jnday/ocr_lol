import os
from wand.image import Image

# Step 0
#
# input: PDF file
# output: one jpg image per page, transformed for ingestion if needed

input_dir = 'input/'
input_filename = 'tables_charts.pdf'
input_path = input_dir + input_filename
output_dir = 'step_0/'
os.makedirs(output_dir, exist_ok=True)
output_filename = input_filename.removesuffix('.pdf') + '.jpg'
output_path = output_dir + output_filename

resolution = 300
colorspace = 'rgb'

# Open given pdf at 300dpi resolution, transform to RGB colorspace, save with filename magic.
# E.g. weick.jpg will expand to weick-0.jpg for each page in the pdf
with Image(filename=input_path, resolution=resolution) as img:
    img.colorspace = colorspace
    img.save(filename=output_path)

