import warnings
import os
import datetime
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'  # 指定使用两张RTX 4090
warnings.filterwarnings('ignore')

from ultralytics import YOLO
# from ultralytics.models import RTDETR

if __name__ == '__main__':
    #  主干 yolov8-MobileNetv4.yaml
    # 注意力  yolov8-MLCA.yaml  yolov8-MobileNetv4-MLCA.yaml
    # 检测头 yolov8-AFPN.yaml  yolov8-MobileNetv4-AFPN.yaml
    # 主干+注意力+检测头 yolov8-MMA.yaml

    # model = RTDETR(model='ultralytics/cfg/models/rt-detr/rt-detr-test.yaml')
    # model = YOLO('ultralytics/cfg/models/v8/yolov8-HGNet1.yaml').load('yolov8n.pt')
    model = YOLO('ultralytics/cfg/models/v8/yolov8-HaarHGNet-l').load('E:/SC/Yolov8/runs/train/yolov8+HGNetv2+WiseIoU/weights/yolov8-HGNet+MIoU-best.pt')
    #model = YOLO('ultralytics/cfg/models/v8/yolov8.yaml').load('yolov8n.pt')
    print("Training model...")

    start_time = datetime.datetime.now()
    print("训练开始时间：", start_time)
    model.train(data='ultralytics/cfg/data.yaml',
                # cache=False,
                cache=True,
                imgsz=640,
                epochs=100,
                batch=16,
                # accumulate=2,  # 梯度累积2次，等效于batch=32
                # close_mosaic=10,
                workers = 16,  # 增加数据加载线程
                # device='0,1',
                device=0, # using SGD

                # optimizer='AdamW', # using,
                optimizer='SGD',

                # lr0=0.02,  # 初始学习率（原0.01 * 2）
                # lrf=0.1,  # 学习率衰减系数
                # resume='', # last.pt path
                amp=False, # close amp
                      # fraction=0.2, #RT-DETR
                project='runs/train',
                name='exp-yolo',
                )
    model.export(format='onnx')  # dynamic=True
    end_time = datetime.datetime.now()
    print("训练结束时间：", end_time)



