# 🚀 YOLOv5 Acne Detection (Training + Evaluation + Result Visualization)
## 📌 Dataset
We use the ACNE04 dataset, a publicly available facial acne detection dataset containing 1450 facial images with bounding box annotations for acne lesions.
you can directly download the yolov5 data format in - 🔗Dataset link: https://universe.roboflow.com/acne-vulgaris-detection/acne04-detection/ 

📂 Format: YOLOv5-ready (images + label .txt files)
  Once downloaded, organize your data as follows:
        ```
        acne04_yolo_dataset/
        ├── train/
        │   ├── images/ (png)
        │   └── labels/ (xml, storing the bounding box coordinate)
        ├── valid/
        │   ├── images/
        │   └── labels/
        └── test/
            ├── images/
            └── labels/
        ```
Each labels/ folder contains .txt annotation files in YOLO format:
<class_id> <x_center> <y_center> <width> <height>, normalized between 0 and 1.

🏋️ Step 1: Training
  Clone YOLOv5:
        ```
        git clone https://github.com/ultralytics/yolov5.git
        cd yolov5
        pip install -r requirements.txt
        ```
        
  Prepare a custom dataset configuration file, make sure the train/val/test dataset path is the same as your own dataset path. e.g., acne.yaml:
       ```
       
        train: dataset/train/images
        val: dataset/valid/images
        test: dataset/test/images
        
        nc: 4
        names: ['nodules and cysts', 'papules', 'pustules', 'whitehead and blackhead']
        
        roboflow:
          workspace: acne-vulgaris-detection
          project: acne04-detection
          version: 5
          license: CC BY 4.0
          url: https://universe.roboflow.com/acne-vulgaris-detection/acne04-detection/dataset/5
    
       ```
   Model selection:
   In our experiments, we selected two YOLOv5 variants for training and evaluation:
  ```
  YOLOv5s (small): lightweight and fast, suitable for limited hardware environments or real-time applications.
  
  YOLOv5m (medium): a balanced model with better accuracy at the cost of slightly more computation.
  ```
  These two models were chosen to explore the trade-off between speed and performance for acne lesion detection.
  All training and evaluation procedures were performed using the same dataset and hyperparameters, with only the model backbone (yolov5s.yaml and yolov5m.yaml) changed accordingly.
  
  To switch between models, simply modify the --cfg and --weights parameters during training.

  Remind!!!!!!! We need to adapt the yolov5s.yaml/yolov5m.yaml to adapt your training set! you should pay attention to the ```nc``` ```depth_multiple``````width_multiple``` and ```anchor```(our project should carefully adapt the anchor since the detect goal is too small comparing to the initial detecting object(car, cat, dog...))

  The backbone yaml include:
      ```
    # Parameters
    nc: 80 # number of classes
    depth_multiple: 1.0 # model depth multiple
    width_multiple: 1.0 # layer channel multiple
    anchors:
      - [10, 13, 16, 30, 33, 23] # P3/8
      - [30, 61, 62, 45, 59, 119] # P4/16
      - [116, 90, 156, 198, 373, 326] # P5/32
    ```
  For example, we adapt the yolov5m.yaml as yolov5m_acne.yaml:

      ```
    # Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
    
    # Parameters
    nc: 4  # 修改类别数为4（与你的数据集匹配）
    depth_multiple: 0.67  # 模型深度倍数（保持v5m的配置）
    width_multiple: 0.75  # 通道宽度倍数（保持v5m的配置）
    anchors:
      - [10, 13, 16, 30, 33, 23]  # P3/8
      - [30, 61, 62, 45, 59, 119]  # P4/16
      - [116, 90, 156, 198, 373, 326]  # P5/32
    
    # YOLOv5 v6.0 backbone（保持不变）
    backbone:
      # [from, number, module, args]
      [
        [-1, 1, Conv, [64, 6, 2, 2]],  # 0-P1/2
        [-1, 1, Conv, [128, 3, 2]],  # 1-P2/4
        [-1, 3, C3, [128]],
        [-1, 1, Conv, [256, 3, 2]],  # 3-P3/8
        [-1, 6, C3, [256]],
        [-1, 1, Conv, [512, 3, 2]],  # 5-P4/16
        [-1, 9, C3, [512]],
        [-1, 1, Conv, [1024, 3, 2]],  # 7-P5/32
        [-1, 3, C3, [1024]],
        [-1, 1, SPPF, [1024, 5]],  # 9
      ]
    
    # YOLOv5 v6.0 head（保持不变）
    head: [
        [-1, 1, Conv, [512, 1, 1]],
        [-1, 1, nn.Upsample, [None, 2, "nearest"]],
        [[-1, 6], 1, Concat, [1]],  # cat backbone P4
        [-1, 3, C3, [512, False]],  # 13
    
        [-1, 1, Conv, [256, 1, 1]],
        [-1, 1, nn.Upsample, [None, 2, "nearest"]],
        [[-1, 4], 1, Concat, [1]],  # cat backbone P3
        [-1, 3, C3, [256, False]],  # 17 (P3/8-small)
    
        [-1, 1, Conv, [256, 3, 2]],
        [[-1, 14], 1, Concat, [1]],  # cat head P4
        [-1, 3, C3, [512, False]],  # 20 (P4/16-medium)
    
        [-1, 1, Conv, [512, 3, 2]],
        [[-1, 10], 1, Concat, [1]],  # cat head P5
        [-1, 3, C3, [1024, False]],  # 23 (P5/32-large)
    
        [[17, 20, 23], 1, Detect, [nc, anchors]],  # Detect(P3, P4, P5)
      ]
    ```
        


  Start training:
  ```
  python train.py --data /data_lg/keru/project/dataset_ACNE04/data.yaml -
                  -cfg /data_lg/keru/project/yolov5/models/yolov5s_ance.yaml
                  --weights /data_lg/keru/project/yolov5/yolov5s.pt
                  --epochs 200 --batch-size 16
  ```
  ```
  python train.py \
  --data /data_lg/keru/project/dataset_ACNE04/data.yaml \
  --cfg /data_lg/keru/project/yolov5/models/yolov5m_ance.yaml \
  --weights /data_lg/keru/project/yolov5/yolov5m.pt \
  --epochs 200 \
  --batch-size 16  # 根据你的GPU显存调整
  ```
🧪 Step 2: Evaluation
After training, evaluate the model using:

bash
Copy
Edit
python val.py --weights runs/train/acne_yolov5/weights/best.pt --data acne.yaml --img 640
This will report metrics like mAP@0.5, Precision, Recall, and F1 score.

👀 Step 3: Inference & Result Visualization
To run inference on test images and visualize results:

bash
Copy
Edit
python detect.py --weights runs/train/acne_yolov5/weights/best.pt --source /path/to/test/images --conf 0.25 --save-txt --save-conf
Predictions will be saved in runs/detect/exp/ by default.

Set --source to a directory, image file, or video file.

Use --save-txt to export YOLO-format predictions, and --save-conf to include confidence scores.

📝 Notes
You can use train.py with different YOLOv5 models: yolov5n, yolov5s, yolov5m, yolov5l, yolov5x depending on your hardware.
