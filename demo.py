import argparse
import time
import torch
from models.yolov3 import yolov3
import cv2
import random
import os
import glob
import math
import numpy as np
from shapely.geometry import Polygon
import shapely
import torch.nn.functional as F

cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if cuda else 'cpu')

parser = argparse.ArgumentParser()

parser.add_argument('-image_folder', type=str, default='./data/test', help='path to images')
# parser.add_argument('-image_folder', type=str, default='/py/YOLOv3-quadrangle/data/samples', help='path to images')
parser.add_argument('-output_folder', type=str, default='result', help='path to outputs')
parser.add_argument('-plot_flag', type=bool, default=True)
parser.add_argument('-txt_out', type=bool, default=False)

parser.add_argument('-cfg', type=str, default='cfg/yolov3.cfg', help='cfg file path')
parser.add_argument('-weights_path', type=str, default='checkpoint/20.pt', help='weight file path')
parser.add_argument('-class_path', type=str, default='data/data.names', help='path to class label file')
parser.add_argument('-conf_thres', type=float, default=0.1, help='object confidence threshold')
parser.add_argument('-nms_thres', type=float, default=0.2, help='iou threshold for non-maximum suppression')
parser.add_argument('-batch_size', type=int, default=1, help='size of the batches')
parser.add_argument('-img_size', type=int, default=608, help='size of each image dimension')
opt = parser.parse_args()
print(opt)

class load_images():  # for inference
	def __init__(self, path, batch_size=1, img_size=416):
		if os.path.isdir(path):
			self.files = sorted(glob.glob('%s/*.png' % path))

		elif os.path.isfile(path):
			self.files = [path]

		self.nF = len(self.files)  # number of image files
		self.nB = math.ceil(self.nF / batch_size)  # number of batches
		self.batch_size = batch_size
		self.height = img_size

		assert self.nF > 0, 'No images found in path %s' % path

	# RGB normalization values
	# self.rgb_mean = np.array([60.134, 49.697, 40.746], dtype=np.float32).reshape((3, 1, 1))
	# self.rgb_std = np.array([29.99, 24.498, 22.046], dtype=np.float32).reshape((3, 1, 1))

	def __iter__(self):
		self.count = -1
		return self

	def __next__(self):
		self.count += 1
		if self.count == self.nB:
			raise StopIteration
		img_path = self.files[self.count]
		# Read image
		o_img = cv2.imread(img_path)  # BGR

		# Padded resize
		img, _, _, _ = resize_square(o_img, height=self.height, color=(127.5, 127.5, 127.5))

		# Normalize RGB
		img = img[:, :, ::-1].transpose(2, 0, 1)
		img = np.ascontiguousarray(img, dtype=np.float32)
		# img -= self.rgb_mean
		# img /= self.rgb_std
		img /= 255.0

		return [img_path], img

def resize_square(img, height=416, color=(0, 0, 0)):  # resize a rectangular image to a padded square
	img = np.array(img)
	shape = img.shape[:2]  # shape = [height, width]
	ratio = float(height) / max(shape)  # ratio  = old / new
	new_shape = [round(shape[0] * ratio), round(shape[1] * ratio)]
	dw = height - new_shape[1]  # width padding
	dh = height - new_shape[0]  # height padding
	top, bottom = dh // 2, dh - (dh // 2)
	left, right = dw // 2, dw - (dw // 2)
	img = cv2.resize(img, (new_shape[1], new_shape[0]), interpolation=cv2.INTER_AREA)  # resized, no border
	return cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color), ratio, dw // 2, dh // 2

def load_classes(path):
	"""
	Loads class labels at 'path'
	"""
	fp = open(path, "r")
	names = fp.read().split("\n")[:-1]
	return names

def plot_one_box(x, img, color=None, label=None, line_thickness=None):  # Plots one bounding box on image img
	tl = line_thickness or round(0.001 * max(img.shape[0:2])) + 1  # line thickness
	color = color or [random.randint(0, 255) for _ in range(3)]

	#c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
	#cv2.rectangle(img, c1, c2, color, thickness=tl)

	cv2.line(img, (int(x[0]), int(x[1])), (int(x[2]), int(x[3])), color, tl)
	cv2.line(img, (int(x[2]), int(x[3])), (int(x[4]), int(x[5])), color, tl)
	cv2.line(img, (int(x[4]), int(x[5])), (int(x[6]), int(x[7])), color, tl)
	cv2.line(img, (int(x[6]), int(x[7])), (int(x[0]), int(x[1])), color, tl)

def bbox_iou_nms(box1, box2):
	cuda = torch.cuda.is_available()
	device = torch.device('cuda:0' if cuda else 'cpu')

	nBox = box2.size()[0]

	iou = torch.zeros(nBox)
	polygon1 = Polygon(box1.view(4,2)).convex_hull
	for i in range(0, nBox):
		polygon2 = Polygon(box2[i,:].view(4,2)).convex_hull
		if polygon1.intersects(polygon2):
			try:
				inter_area = polygon1.intersection(polygon2).area
				union_area = polygon1.union(polygon2).area
				iou[i] =  inter_area / union_area
			except shapely.geos.TopologicalError:
				print('shapely.geos.TopologicalError occured, iou set to 0')
				iou[i] = 0

	return iou.to(device)
def non_max_suppression(prediction, cls_thres=0.5, nms_thres=0.4):
	prediction = prediction.cpu()

	output = [None for _ in range(len(prediction))]

	for image_i, pred in enumerate(prediction):
		class_prob, class_pred = torch.max(F.softmax(pred[:, 9:], 1), 1)

		v = (class_prob > cls_thres).numpy()
		v = v.nonzero()

		pred = pred[v]
		class_prob = class_prob[v]
		class_pred = class_pred[v]

		# If none are remaining => process next image
		nP = pred.shape[0]
		if not nP:
			continue

		detections = torch.cat((pred[:, :9], class_prob.float().unsqueeze(1), class_pred.float().unsqueeze(1)), 1)
		# Iterate through all predicted classes
		unique_labels = detections[:, -1].cpu().unique()
		if prediction.is_cuda:
			unique_labels = unique_labels.cuda()

		for c in unique_labels:
			detections_class = detections[detections[:, -1] == c]
			# Sort through confidence in one class
			_, conf_sort_index = torch.sort(detections_class[:, 8], descending=True)
			detections_class = detections_class[conf_sort_index]

			max_detections = []

			while detections_class.shape[0]:
				# Get detection with highest confidence and save as max detection
				max_detections.append(detections_class[0].unsqueeze(0))
				# Stop if we're at the last detection
				if len(detections_class) == 1:
					break
				# Get the IOUs for all boxes with lower confidence
				ious = bbox_iou_nms(max_detections[-1].squeeze(0)[0:8], detections_class[1:][:, 0:8])

				# Remove detections with IoU >= NMS threshold
				detections_class = detections_class[1:][ious < nms_thres]

			if len(max_detections) > 0:
				max_detections = torch.cat(max_detections).data
				# Add max detections to outputs
				output[image_i] = max_detections if output[image_i] is None else torch.cat(
					(output[image_i], max_detections))

	return output

def detect(opt):
	os.system('rm -rf ' + opt.output_folder)
	os.makedirs(opt.output_folder, exist_ok=True)

	# Load model
	model = yolov3()

	weights_path = opt.weights_path


	checkpoint = torch.load(weights_path)

	model.load_state_dict(checkpoint)
	del checkpoint
	model.to(device).eval()

	# Set Dataloader
	classes = load_classes(opt.class_path)  # Extracts class labels from file
	dataloader = load_images(opt.image_folder, batch_size=opt.batch_size, img_size=opt.img_size)

	imgs = []  # Stores image paths
	img_detections = []  # Stores detections for each image index
	prev_time = time.time()
	for batch_i, (img_paths, img) in enumerate(dataloader):
		print(batch_i, img.shape, end=' ')

		# Get detections
		with torch.no_grad():
			chip = torch.from_numpy(img).unsqueeze(0).to(device)
			_,pred = model(chip)
			pred = pred.unsqueeze(0)
			print(pred.shape)
			pred = pred[pred[:, :, 8] > opt.conf_thres]
			if len(pred) > 0:
				detections = non_max_suppression(pred.unsqueeze(0), 0.1, opt.nms_thres)
				img_detections.extend(detections)
				imgs.extend(img_paths)

		print('Batch %d... (Done %.3f s)' % (batch_i, time.time() - prev_time))
		prev_time = time.time()

	# Bounding-box colors
	color_list = [[random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)] for _ in range(len(classes))]

	if len(img_detections) == 0:
		return

	# Iterate through images and save plot of detections
	for img_i, (path, detections) in enumerate(zip(imgs, img_detections)):
		print("image %g: '%s'" % (img_i, path))

		img = cv2.imread(path)

		# The amount of padding that was added
		pad_x = max(img.shape[0] - img.shape[1], 0) * (opt.img_size / max(img.shape))
		pad_y = max(img.shape[1] - img.shape[0], 0) * (opt.img_size / max(img.shape))
		# Image height and width after padding is removed
		unpad_h = opt.img_size - pad_y
		unpad_w = opt.img_size - pad_x

		# Draw bounding boxes and labels of detections
		if detections is not None:
			unique_classes = detections[:, -1].cpu().unique()

			bbox_colors = random.sample(color_list, len(unique_classes))

			# write results to .txt file
			results_img_path = os.path.join(opt.output_folder, path.split('/')[-1])
			results_txt_path = results_img_path.replace('png', 'txt')
			if os.path.isfile(results_txt_path):
				os.remove(results_txt_path)

			for i in unique_classes:
				n = (detections[:, -1].cpu() == i).sum()
				print('%g %ss' % (n, classes[int(i)]))

			for P1_x, P1_y, P2_x, P2_y, P3_x, P3_y, P4_x, P4_y, conf, cls_conf, cls_pred in detections:
				P1_y = max((((P1_y - pad_y // 2) / unpad_h) * img.shape[0]).round().item(), 0)
				P1_x = max((((P1_x - pad_x // 2) / unpad_w) * img.shape[1]).round().item(), 0)
				P2_y = max((((P2_y - pad_y // 2) / unpad_h) * img.shape[0]).round().item(), 0)
				P2_x = max((((P2_x - pad_x // 2) / unpad_w) * img.shape[1]).round().item(), 0)
				P3_y = max((((P3_y - pad_y // 2) / unpad_h) * img.shape[0]).round().item(), 0)
				P3_x = max((((P3_x - pad_x // 2) / unpad_w) * img.shape[1]).round().item(), 0)
				P4_y = max((((P4_y - pad_y // 2) / unpad_h) * img.shape[0]).round().item(), 0)
				P4_x = max((((P4_x - pad_x // 2) / unpad_w) * img.shape[1]).round().item(), 0)
				# write to file
				if opt.txt_out:
					with open(results_txt_path, 'a') as f:
						f.write(('%s %.2f %g %g %g %g %g %g %g %g  \n') % \
							(classes[int(cls_pred)], cls_conf * conf, P1_x, P1_y, P2_x, P2_y, P3_x, P3_y, P4_x, P4_y ))
				if opt.plot_flag:
					# Add the bbox to the plot
					label = '%s %.2f' % (classes[int(cls_pred)], conf)
					color = bbox_colors[int(np.where(unique_classes == int(cls_pred))[0])]
					plot_one_box([P1_x, P1_y, P2_x, P2_y, P3_x, P3_y, P4_x, P4_y], img, label=None, color=color)
			'''
			if opt.plot_flag:
				cv2.imshow(path.split('/')[-1], img)
				cv2.waitKey(0)
				cv2.destroyAllWindows()
			'''
		if opt.plot_flag:
			# Save generated image with detections
			cv2.imwrite(results_img_path.replace('.bmp', '.jpg').replace('.tif', '.jpg'), img)

if __name__ == '__main__':
	torch.cuda.empty_cache()
	detect(opt)