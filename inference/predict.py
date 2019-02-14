from openvino.inference_engine import IENetwork, IEPlugin
import numpy as np
import cv2
import json

with open("PATHS.json", "r") as f:
	paths = json.load(f)
print(json.dumps(paths, sort_keys=True, indent=4))

def pre_process_image(imagePath):
    r = cv2.imread(imagePath+"_red.png", cv2.IMREAD_GRAYSCALE)
    g = cv2.imread(imagePath+"_green.png", cv2.IMREAD_GRAYSCALE)
    b = cv2.imread(imagePath+"_blue.png", cv2.IMREAD_GRAYSCALE)
    n, c, h, w    = [1, 3, 512, 512] 
    image = np.stack((r, g, b), -1)
    image = np.rollaxis(image, 2, 0)
    image = np.expand_dims(image, axis=0)
    
    return image, image, imagePath


# Devices: GPU (intel), CPU, MYRIAD
plugin = IEPlugin("CPU")
# Read IR
net = IENetwork.from_ir(model=paths['MODEL_DIR']+"sn.xml", weights=paths['MODEL_DIR']+"sn.bin")
assert len(net.inputs.keys()) == 1
assert len(net.outputs) == 1 
input_blob = next(iter(net.inputs))
out_blob = next(iter(net.outputs))
# Load network to the plugin
exec_net = plugin.load(network=net)
del net
# Run inference
import time
import pandas as pd
import tqdm
s = time.time()
data = pd.read_csv(paths['COMP_DATA_DIR']+"sample_submission.csv")
count = 0
predicted = []
tot_time = 0
l = len(data)
count = 0
for ID in tqdm.tqdm(data['Id']):
	image, processedImg, imagePath = pre_process_image(paths['COMP_DATA_DIR']+"test/"+ID)
	s1 = time.time()
	res = exec_net.infer(inputs={input_blob: processedImg})
	s2 = time.time()
	tot_time += s2-s1
	# Access the results and get the index of the highest confidence score
	res = res['dense_1/Sigmoid']
#	idx = np.argsort(res[0])[-1]
	predicted.append(" ".join([str(i) for i in np.arange(28)[res[0]>0.23]]))
print(time.time()-s)
print(tot_time)
data['Predicted'] = predicted
data.to_csv(paths['OUTPUT_DIR']+'submission.csv', index=False)

