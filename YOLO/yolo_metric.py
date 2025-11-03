from ultralytics import YOLO

if __name__ == '__main__':

    model = YOLO("YOLO\DOTA.pt" )
    metrics = model.val(data=r'E:\project\yolo_agent\data\DOTA\all.yaml') # 使用验证集评估
    print(metrics.box.map)  # 输出mAP指标;
    # source = r"E:\project\yolo_agent\data\DOTA\images\val"
    # results = model.predict(source,save=True,line_width=1,save_dir="E:\project\yolo_agent\outputs\yolo_predictions")  

