import cv2, glob
from imread_from_url import imread_from_url 

from src import InferenceEngine
from src.utils import select_for_labeling

model_path = "/home/daniele/Modelli/fcn.onnx"

# Initialize semantic segmentator
segmentator = InferenceEngine(model_path)

# single image from the web
# img = imread_from_url("https://westgate-global.com/wp-content/uploads/2023/03/Westgate_0099_Easiguard_handrail.jpg")
# single local image
# img = cv2.imread("images/20220513_093448_000.jpg")

# multiple local images
for image_path in glob.glob("dataset/*/*.jpg"):
    img = cv2.imread(image_path)
    
    # Update semantic segmentator
    seg_map = segmentator(img)
    
    select_for_labeling(seg_map, image_path)
    
    # To Show
    # combined_img = segmentator.draw_segmentation(img, alpha=0)
    # cv2.namedWindow("Output", cv2.WINDOW_NORMAL)
    # cv2.imshow("Output", combined_img)
    # cv2.waitKey(0)
