
import json
from flask import Flask, request, jsonify
from func.crop_image import crop_image
from func.crop_face import crop_face

from func.processing_tools import url_to_image

app = Flask(__name__)

@app.route('/api/v1/extract_face', methods =['POST'])
def face_api():
    data = json.loads(request.data)
    img_url = data['img_url']
    ima = url_to_image(img_url)
    

    # weight of crop_image
    crop_img_model_path = './weight/idcard_yolo/yolov7_4000epoch_n.pt'

    # weight of detect_face
    crop_face_model_path = './weight/idcard_extract_face/face_detect.pt'

#------------------------------------------------------------------------------------------------


# output
    a = crop_image(ima,crop_img_model_path)
    loc_face = crop_face(a, crop_face_model_path)  
    loc_face = loc_face.tolist()
    return jsonify(loc_face)
app.run(debug=True)