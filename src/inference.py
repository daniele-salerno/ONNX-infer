import cv2
import numpy as np
import onnxruntime
from imread_from_url import imread_from_url
import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))


from src.utils import util_draw_seg

class InferenceEngine():

	def __init__(self, model_path):

		# Initialize model
		self.initialize_model(model_path)

	def __call__(self, image):
		return self.estimate_segmentation(image)

	def initialize_model(self, model_path):

		self.session = onnxruntime.InferenceSession(model_path, 
													providers=['CUDAExecutionProvider', 
															   'CPUExecutionProvider'])

		# Get model info
		self.get_input_details()
		self.get_output_details()

	def estimate_segmentation(self, image):

		input_tensor = self.prepare_input(image)

		# Perform inference on the image
		outputs = self.inference(input_tensor)

		# Process output data
		self.seg_map = self.process_output(outputs)

		return self.seg_map

	def prepare_input(self, image):

		self.img_height, self.img_width = image.shape[:2]

		input_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

		# Resize input image
		input_img = cv2.resize(input_img, (self.input_width,self.input_height))  

		# Scale input pixel values to -1 to 1
		# mean=[0.485, 0.456, 0.406]
		# std=[0.229, 0.224, 0.225]
		# input_img = ((input_img/255.0 - mean) / std)

		mean=[123.675, 116.28, 103.53]
		std=[58.395, 57.12, 57.375]

		input_img = ((input_img - mean) / std)
		input_img = input_img.transpose(2, 0, 1)
		input_tensor = input_img[np.newaxis,:,:,:].astype(np.float32)   

		return input_tensor

	def inference(self, input_tensor):

		outputs = self.session.run(self.output_names, {self.input_names[0]: input_tensor})

		return outputs

	def process_output(self, outputs): 

		#return np.squeeze(np.argmax(outputs[0], axis=1))
		return np.squeeze(outputs[0]) # from (1,1,512,512) -> (512,512)

	def draw_segmentation(self, image, alpha = 0.5):

		return util_draw_seg(self.seg_map, image, alpha)

	def get_input_details(self):

		model_inputs = self.session.get_inputs()
		self.input_names = [model_inputs[i].name for i in range(len(model_inputs))]

		self.input_shape = model_inputs[0].shape
		self.input_height = self.input_shape[2]
		self.input_width = self.input_shape[3]

	def get_output_details(self):

		model_outputs = self.session.get_outputs()
		self.output_names = [model_outputs[i].name for i in range(len(model_outputs))]

if __name__ == '__main__':
    
	onnx_model_path = "./models/stdc_512.onnx"

	# Initialize semantic segmentator
	segmentator = InferenceEngine(onnx_model_path)

	img = imread_from_url("https://westgate-global.com/wp-content/uploads/2023/03/Westgate_0099_Easiguard_handrail.jpg")

	# Update semantic segmentator
	seg_map = segmentator(img)

	combined_img = segmentator.draw_segmentation(img, alpha=0.5)
	cv2.namedWindow("Output", cv2.WINDOW_NORMAL)
	cv2.imshow("Output", combined_img)
	cv2.waitKey(0)
