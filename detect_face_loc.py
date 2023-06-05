import cv2

from func.crop_image import crop_image
from func.crop_face import crop_face

# insert path here

#------------------------------------------------------------------------------------------------

# input image
img = cv2.imread('./test_image/7.jpg') 
image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# weight of crop_image
crop_img_model_path = './weight/idcard_yolo/yolov7_4000epoch_n.pt'

# weight of detect_word
crop_face_model_path = './weight/idcard_extract_face/face_detect.pt'

#------------------------------------------------------------------------------------------------


# output
inp = crop_image(image,crop_img_model_path)
loc_face = crop_face(inp, crop_face_model_path)
print(type(loc_face))