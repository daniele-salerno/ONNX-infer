import numpy as np
import cv2, os, shutil
from csv import DictWriter

# list of column names in csv
field_names = ['NAME', 'RANK']

dir_path = os.path.dirname(os.path.abspath(__file__))
ade20k_info_path = dir_path + "/ade20k_label_colors_edited.txt"

def read_ade20k_info(info_path=ade20k_info_path):
	with open(info_path) as fp:
		lines = fp.readlines()

		labels = [line[:-1].replace(';', ',').split(',')[0] for line in lines]
		colors = np.array([line[:-1].replace(';', ',').split(',')[-3:] for line in lines]).astype(int)

	return colors, labels

colors, labels = read_ade20k_info()
	
def util_draw_seg(seg_map, image, alpha = 0.5):

	# Convert segmentation prediction to colors
	color_segmap = cv2.resize(image, (seg_map.shape[1], seg_map.shape[0]))
	color_segmap[seg_map>=0] = colors[seg_map[seg_map>=0]]

	# Resize to match the image shape
	color_segmap = cv2.resize(color_segmap, (image.shape[1],image.shape[0]))

	# Fuse both images
	if(alpha == 0):
		combined_img = np.hstack((image, color_segmap))
	else:
		combined_img = cv2.addWeighted(image, alpha, color_segmap, (1-alpha),0)

	return combined_img

def select_for_labeling(seg_map, img_path, treshold=1000):
    
    os.makedirs("images/to_label/", exist_ok=True)
    unique_classes, counts = np.unique(seg_map, return_counts=True)

    pixel_barriers = 0
    pixel_ped_crossing = 0
    
    # Displaying the count of each number/class
    for class_num, count in zip(unique_classes, counts):
        print(f"Number/Class {class_num} occurs {count} times.")
        
        pixel_barriers = count if class_num == 1 else int(pixel_barriers)
        pixel_ped_crossing = count if class_num == 3 else int(pixel_ped_crossing)
        
    if pixel_barriers + pixel_ped_crossing >= treshold:
        print(f"moving {img_path}")
        shutil.copy2(img_path, "images/to_label/")
        
        dict = {'NAME': os.path.basename(img_path), 'RANK': pixel_barriers + pixel_ped_crossing}
        
        with open('images/to_label/000event.csv', 'a') as f_object:
            dictwriter_object = DictWriter(f_object, fieldnames=field_names)
            dictwriter_object.writerow(dict)
            f_object.close()
