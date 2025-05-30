import os
import sys
import cv2
import torch
import glob
from pathlib import Path

# Настройка пути к YOLOv5
FILE = os.path.abspath(__file__)
ROOT = os.path.dirname(FILE)
YOLOV5_PATH = os.path.join(ROOT, 'yolov5')
sys.path.append(YOLOV5_PATH)

from yolov5.models.common import DetectMultiBackend
from yolov5.utils.general import non_max_suppression, scale_coords
from yolov5.utils.torch_utils import select_device

def load_images_from_folder(folder, img_size=640):
    image_paths = glob.glob(os.path.join(folder, '*.*'))
    images = []
    for path in image_paths:
        img0 = cv2.imread(path)
        if img0 is None:
            continue
        img = cv2.resize(img0, (img_size, img_size))
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB
        img = torch.from_numpy(img).float() / 255.0
        images.append((path, img.unsqueeze(0), img0))
    return images

def detect(source='images', weights='best.pt', imgsz=640, conf_thres=0.25):
    device = select_device('')
    model = DetectMultiBackend(weights, device=device)
    model.eval()

    os.makedirs('outputs', exist_ok=True)
    data = load_images_from_folder(source, img_size=imgsz)

    for path, img, im0s in data:
        img = img.to(device)

        pred = model(img, augment=False)
        pred = non_max_suppression(pred, conf_thres, 0.45, classes=None, agnostic=False)

        for det in pred:
            if len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0s.shape).round()
                for i, (*xyxy, conf, cls) in enumerate(det):
                    x1, y1, x2, y2 = map(int, xyxy)
                    plate = im0s[y1:y2, x1:x2]

                    filename = os.path.basename(path)
                    name, _ = os.path.splitext(filename)
                    save_path = f'outputs/{name}_plate_{i}.jpg'
                    cv2.imwrite(save_path, plate)
                    print(f'[+] Plate saved to {save_path}')

if __name__ == '__main__':
    detect()
