from ultralytics import YOLO

if __name__ == '__main__':
    results = YOLO("9e.pt").val(data="oldaug/data.yaml", batch=16, name="val-9e-oldaug")