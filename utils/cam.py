# simple implementation of CAM in PyTorch for the networks such as ResNet, DenseNet, SqueezeNet, Inception

from PIL import Image
import numpy as np
import cv2


def returnCAM(feature_conv, weight_softmax, class_idx):
    # generate the class activation maps upsample to 320x320
    size_upsample = (240, 240)
    bz, nc, h, w = feature_conv.shape
    idx = class_idx
    cam = weight_softmax[idx].dot(feature_conv.reshape((nc, h*w)))
    cam = cam.reshape(h, w)
    cam = np.maximum(cam, 0)
    cam = cam - np.min(cam)
    cam = cam / np.max(cam)
    cam[0][-1] = cam[0][0]
    cam[-1][-1] = cam[-1][0]
    return cam

def get_CAM(model, finalconv_name, mid_feature_name, img, pic):
    # hook the feature extractor
    features_blobs = []
    mid_features = []
    def hook_feature(module, input, output):
        features_blobs.append(output.data.cpu().numpy())
    def hook_mid_feature(module, input, output):
        mid_features.append(output.data)
    model._modules.get(finalconv_name).register_forward_hook(hook_feature)
    model._modules.get(mid_feature_name).register_forward_hook(hook_mid_feature)
    # get the softmax weight
    params = list(model.parameters())
    weight_softmax = np.squeeze(params[-2].data.numpy())
    logit = model(pic)
    index = np.argmax(logit.cpu().data.numpy())
    CAM = returnCAM(features_blobs[0], weight_softmax, index)
    
    height, width, _ = img.shape
    img = np.float32(img) / 255
    heatmap = cv2.applyColorMap(cv2.resize(np.uint8(255*CAM),(width, height)), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    cam_result = np.uint8(255*cam)
    # import matplotlib.pyplot as plt
    # plt.imshow(cam_result)
    # plt.show()
    return cam_result, logit, mid_features[0]