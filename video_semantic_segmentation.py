import cv2
#import pafy

from src import InferenceEngine

# Initialize video
cap = cv2.VideoCapture("video/20220317_164108.mp4")

# videoUrl = 'https://youtu.be/mPkBrray4mU?si=DDq0K8AKBtAjiM_z'
# videoPafy = pafy.new(videoUrl)
# print(videoPafy.streams)
# cap = cv2.VideoCapture(videoPafy.streams[-1].url)
start_time = 30 # skip first {start_time} seconds
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

# Initialize semantic segmentator
model_path = "models/stdc_512.onnx"
segmentator = InferenceEngine(model_path)

cv2.namedWindow("Semantic Sementation", cv2.WINDOW_NORMAL)	
while cap.isOpened():

	# Press key q to stop
	if cv2.waitKey(1) == ord('q'):
		break

	try:
		# Read frame from the video
		ret, frame = cap.read()
		if not ret:	
			break
	except:
		continue
	
	# Update semantic segmentator
	seg_map = segmentator(frame)
	combined_img = segmentator.draw_segmentation(frame, alpha=0.5)
	cv2.imshow("Semantic Sementation", combined_img)
