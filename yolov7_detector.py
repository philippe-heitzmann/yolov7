'''Example run:
yolo = YOLOv7(weights='yolov7.pt', source='inference/images', img_size=640, device='', half=True,
              trace=True, augment=True, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic_nms=False)
yolo.detect()
'''

import torch

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, select_device, non_max_suppression, apply_classifier
from utils.torch_utils import TracedModel, load_classifier

class YOLOv7Detector:
    def __init__(self, weights='/weights/yolov7.pt', source='inference/images', img_size=640, device='', half=True,
                 trace=True, augment=True, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic_nms=False, classifier=False):
        self.weights = weights
        self.source = source
        self.img_size = img_size
        self.device = select_device(device)
        self.half = half
        self.trace = trace
        self.augment = augment
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.classes = classes
        self.agnostic_nms = agnostic_nms
        self.webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
        self.model, self.stride, self.names = self.load_model()
        self.classifier = classifier
        if self.classifier:
            self._load_classifier()

    def load_model(self):
        model = attempt_load(self.weights, map_location=self.device)  # load FP32 model
        stride = int(model.stride.max())  # model stride
        self.img_size = check_img_size(self.img_size, s=stride)  # check img_size
        if self.trace:
            model = TracedModel(model, self.device, self.img_size)
        if self.half:
            model.half()  # to FP16
        names = model.module.names if hasattr(model, 'module') else model.names
        return model, stride, names
    
    def _load_classifier(self):
        self.modelc = load_classifier(name='resnet101', n=2)  # initialize
        self.modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=self.device)['model']).to(self.device).eval()

    def get_dataset(self):
        if self.webcam:
            dataset = LoadStreams(self.source, img_size=self.img_size, stride=self.stride)
        else:
            dataset = LoadImages(self.source, img_size=self.img_size, stride=self.stride)
        return dataset

    def detect(self, classify = False):
        dataset = self.get_dataset()
        if self.device.type != 'cpu':
            self.model(torch.zeros(1, 3, self.img_size, self.img_size).to(self.device).type_as(next(self.model.parameters())))  # run once
        for path, img, im0s, vid_cap in dataset:
            img = torch.from_numpy(img).to(self.device)
            img = img.half() if self.half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            with torch.no_grad():   
                pred = self.model(img, augment=self.augment)[0]

            # Apply NMS
            pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, classes=self.classes, agnostic=self.agnostic_nms)
            # Pass pred to second-stage classifier
            if classify:
                pred = self.classify(pred, self.modelc, img, im0s)
        return pred

    def classify(self, classifier, pred, img, im0s):
        # Apply second stage classifier here.
        pred = apply_classifier(pred, classifier, img, im0s)
        return pred
