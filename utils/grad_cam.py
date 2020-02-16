import torch
from torch.autograd import Variable
from torch.autograd import Function
from torchvision import models
from torchvision import utils
import cv2
import torch.nn.functional as F
import sys
import numpy as np
import argparse

class FeatureExtractor():
    """ Class for extracting activations and 
    registering gradients from targetted intermediate layers """
    def __init__(self, model, target_layers, mode):
        self.model = model
        self.mode = mode
        self.target_layers = target_layers
        self.gradients = []

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def __call__(self, x):
        outputs = []
        self.gradients = []
        for name, module in self.model._modules.items():
            if self.mode == 'efficientnet':
                if name in self.target_layers:
                    x = x.detach()
                if name == '_blocks':
                    blocks = module._modules.items()
                    for b in blocks:
                        x = b[1](x)
                else:
                    x = module(x)
            else:
                x = module(x)
            if name in self.target_layers:
                x.register_hook(self.save_gradient)
                outputs += [x]
                break
        return outputs, x #[layer4_output], layer4_output

class ModelOutputs():
	""" Class for making a forward pass, and getting:
	1. The network output.
	2. Activations from intermeddiate targetted layers.
	3. Gradients from intermeddiate targetted layers. """
	def __init__(self, model, target_layers, mode='resnet'):
		self.model = model
		self.mode = mode
		if mode == 'resnet' or mode == 'efficientnet':
			self.feature_extractor = FeatureExtractor(self.model, target_layers, mode) #修改self.model.features
		else:
			self.feature_extractor = FeatureExtractor(self.model.features, target_layers, mode)
	def get_gradients(self):
		return self.feature_extractor.gradients

	def __call__(self, x):
		target_activations, output  = self.feature_extractor(x)
		# output = output.view(output.size(0), -1)

		if self.mode == 'vgg16':
			output = F.adaptive_avg_pool2d(output, (1, 1)).view(output.size(0), -1)
		else:
			if self.mode == 'efficientnet':
				output = self.model._avg_pooling(output)
			else:
				output = self.model.avgpool(output)
			output = output.view(output.size(0), -1)
			mid_feature = output
			# print('OOOO', output)
		# output = self.model.classifier(output)
		if self.mode =='resnet':
			output = self.model.fc(output)
			# output = self.model._fc(output)
			# print('CAM output:', output)
		elif self.mode == 'efficientnet':
			output = self.model._fc(output)
		else:
			output = self.model.classifier(output)
		return target_activations, output, mid_feature


def show_cam_on_image(img, mask):
	heatmap = cv2.applyColorMap(np.uint8(255*mask), cv2.COLORMAP_JET)
	heatmap = np.float32(heatmap) / 255
	cam = heatmap + np.float32(img)
	cam = cam / np.max(cam)
	return np.uint8(255 * cam)
	# cv2.imwrite(save_path, np.uint8(255 * cam))

class GradCam:
	def __init__(self, model, target_layer_names, use_cuda, mode='resnet'):
		if isinstance(model, torch.nn.DataParallel):
			self.model = model.module
		else:
			self.model = model
		self.mode = mode
		self.model.eval()
		self.cuda = use_cuda
		if self.cuda:
			self.model = self.model.cuda()
		self.extractor = ModelOutputs(self.model, target_layer_names, mode=mode)


	def forward(self, input):
		return self.model(input) 

	def __call__(self, input, index = None):
		if self.cuda:
			features, output, mid_feature = self.extractor(input.cuda())
		else:
			features, output, mid_feature = self.extractor(input)

		if index == None:
			index = np.argmax(output.cpu().data.numpy())

		one_hot = np.zeros((1, output.size()[-1]), dtype = np.float32)
		one_hot[0][index] = 1
		one_hot = Variable(torch.from_numpy(one_hot), requires_grad = True)
		if self.cuda:
			one_hot = torch.sum(one_hot.cuda() * output)
		else:
			one_hot = torch.sum(one_hot * output)
		if self.mode == 'resnet' or self.mode == 'efficientnet':
			self.model.zero_grad()
		else:
			self.model.features.zero_grad()
			self.model.classifier.zero_grad()
		one_hot.backward(retain_graph=True)

		grads_val = self.extractor.get_gradients()[-1].cpu().data.numpy()

		target = features[-1]
		target = target.cpu().data.numpy()[0, :]

		weights = np.mean(grads_val, axis = (2, 3))[0, :]
		cam = np.zeros(target.shape[1 : ], dtype = np.float32)

		for i, w in enumerate(weights):
			cam += w * target[i, :, :]

		cam = np.maximum(cam, 0)
		cam = cv2.resize(cam, (320, 320))
		cam = cam - np.min(cam)
		cam = cam / np.max(cam)
		return cam, output, mid_feature


def get_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--use-cuda', action='store_true', default=False,
	                    help='Use NVIDIA GPU acceleration')
	parser.add_argument('--image-path', type=str, default='1.jpg',
	                    help='Input image path')
	args = parser.parse_args()
	args.use_cuda = args.use_cuda and torch.cuda.is_available()
	if args.use_cuda:
		print("Using GPU for acceleration")
	else:
		print("Using CPU for computation")

	return args

'''
样例测试
if __name__ == '__main__':


	args = get_args()
	print('Debug')

	# Can work with any model, but it assumes that the model has a
	# feature method, and a classifier method,
	# as in the VGG models in torchvision.
	# grad_cam = GradCam(model = models.densenet121(pretrained=False), \
	# 				target_layer_names = ["norm5"], use_cuda=args.use_cuda, mode='densenet')
	# grad_cam = GradCam(model=models.vgg16(pretrained=True), \
	# 				   target_layer_names=["30"], use_cuda=args.use_cuda, mode='vgg16')
	import torch.nn as nn
	from eff_model import efficientnet_b5
	from PIL import Image
	import torchvision.transforms as transforms
	normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
	transforms_ = transforms.Compose([
		transforms.Resize(320),
		transforms.CenterCrop(320),
		#ScaleResize((320, 320)),
		transforms.ToTensor(),
		normalize
	])
	img = np.array(Image.open(args.image_path).convert('RGB'))
	pic = Image.fromarray(img)
	pic = transforms_(pic)
	pic = pic.unsqueeze(0)

	pic = torch.Tensor(pic)
	model = efficientnet_b5(pretrained=False)
	in_features = model._fc.in_features
	model._fc = nn.Linear(in_features, 57)
	grad_cam = GradCam(model = model, \
					target_layer_names = ["_bn1"], use_cuda=args.use_cuda, mode='efficientnet')

	# img = cv2.imread(args.image_path, 1)
	img = np.float32(cv2.resize(img, (320, 320))) / 255
	# input = preprocess_image(img)
	input = pic
	# If None, returns the map for the highest scoring category.
	# Otherwise, targets the requested index.
	target_index = None

	mask,_,_ = grad_cam(input, target_index)

	cam = show_cam_on_image(img, mask)
	import matplotlib.pyplot as plt 
	plt.imshow(cam)
	plt.show()
	print()

'''
