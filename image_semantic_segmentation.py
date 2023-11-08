import cv2
from imread_from_url import imread_from_url

from src import InferenceEngine

model_path = "models/stdc_512.onnx"

# Initialize semantic segmentator
segmentator = InferenceEngine(model_path)

img = imread_from_url("https://westgate-global.com/wp-content/uploads/2023/03/Westgate_0099_Easiguard_handrail.jpg")
#img = cv2.imread("images/20220513_093448_000.jpg")

# Update semantic segmentator
seg_map = segmentator(img)

combined_img = segmentator.draw_segmentation(img, alpha=0)
cv2.namedWindow("Output", cv2.WINDOW_NORMAL)
cv2.imshow("Output", combined_img)
cv2.waitKey(0)
