
from urllib.request import urlopen

import sys

from .helper import *

model = get_model_from_az_storage()

def predict_image_from_url(image_url):
    with urlopen(image_url) as test_image:
        try:
            img_tensor = convert_image_to_tensor(model, test_image)
            prediction = get_model_prediction(model, img_tensor)

            response = {'predictedTagName': prediction}
        except:
            response = {'error' : 'image url is invalid'}

        return response

if __name__ == '__main__':
    predict_image_from_url(sys.argv[1])