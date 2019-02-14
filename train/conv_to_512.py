import multiprocessing as mt
from PIL import Image
from PIL import ImageFile
import cv2
import glob
import tqdm
import numpy as np

def process_file(fn):
	colour = fn.split("/")[-1].split(".")[0].split("_")[-1]
	if colour == 'yellow':
		return
	img = cv2.imread(fn)
	if img.data == None:
		print("defective image {}".format(fn))
		return
	ofn = "512_images/"+fn.split("/")[-1].split(".")[0]+".png"
	img = cv2.resize(img, (512,512))[:,:,clr_idx[colour]]
	cv2.imwrite(ofn, img)



if __name__ == '__main__':
	clr_idx = { 'blue':0, 'green':1, 'red':2} 
	pool = mt.Pool(processes=4)

	fnl = glob.glob('hpa_site_data/*_red.jpg')
	queue_list = []
	for fn in tqdm.tqdm(fnl):
		process_file(fn)

	print("finished red")
	
	fnl = glob.glob('hpa_site_data/*_green.jpg')
	queue_list = []
	for fn in tqdm.tqdm(fnl):
		process_file(fn)
	print("finished green")
	
	
	fnl = glob.glob('hpa_site_data/*_blue.jpg')
	queue_list = []
	for fn in tqdm.tqdm(fnl):
		process_file(fn)
	print("finished blue")
	
