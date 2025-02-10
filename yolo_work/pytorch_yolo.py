#pip install opencv-python
#conda install pandas
# I also manually removed @np._no_nep50_warning() from /ssd1/htalendr/miniconda3/envs/env1/lib/python3.12/site-packages/numpy/testing/_private/utils.py

import torch

import numpy as np
import requests
import torch
from PIL import Image

# Model
import sys
sys.path.append('/ssd1/htalendr/yolov5')
# model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model = torch.hub.load('/ssd1/htalendr/yolov5', 'yolov5s', pretrained=True, source='local')

# Images

def exif_transpose(image):
    """
    Transpose a PIL image accordingly if it has an EXIF Orientation tag.
    Inplace version of https://github.com/python-pillow/Pillow/blob/master/src/PIL/ImageOps.py exif_transpose().

    :param image: The image to transpose.
    :return: An image.
    """
    return image
    # TODO try below if issues
    # exif = image.getexif()
    # orientation = exif.get(0x0112, 1)  # default 1
    # if orientation > 1:
    #     method = {
    #         2: Image.FLIP_LEFT_RIGHT,
    #         3: Image.ROTATE_180,
    #         4: Image.FLIP_TOP_BOTTOM,
    #         5: Image.TRANSPOSE,
    #         6: Image.ROTATE_270,
    #         7: Image.TRANSVERSE,
    #         8: Image.ROTATE_90,
    #     }.get(orientation)
    #     if method is not None:
    #         image = image.transpose(method)
    #         del exif[0x0112]
    #         image.info["exif"] = exif.tobytes()
    # return image



im = Image.open(requests.get('https://ultralytics.com/images/zidane.jpg', stream=True).raw)
im = np.asarray(exif_transpose(im))
print("DIMS:", im.shape)
imgs = []
imgs.append(im)
# imgs = ['https://ultralytics.com/images/zidane.jpg']  # batch of images

# Inference
results = model(imgs)

# Results
results.print()
results.save()  # or .show()

results.xyxy[0]  # img1 