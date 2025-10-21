class DOTAAdapter:
    def __init__(self, original_pipeline):
        self.pipeline = original_pipeline
        self.dota_classes = ['plane', 'ship', 'storage-tank', 'baseball-diamond', 
                           'tennis-court', 'basketball-court', 'ground-track-field',
                           'harbor', 'bridge', 'large-vehicle', 'small-vehicle', 
                           'helicopter', 'roundabout', 'soccer-ball-field', 'swimming-pool']
    
    def adapt_query_for_dota(self, query):
        """将通用查询适配到DOTA类别"""
        query_mapping = {
            "车辆": "large-vehicle and small-vehicle",
            "交通工具": "plane, ship, helicopter, large-vehicle, small-vehicle",
            "运动场地": "baseball-diamond, tennis-court, basketball-court, ground-track-field, soccer-ball-field"
        }
        return query_mapping.get(query, query)
    
    def preprocess_dota_image(self, image):
        """DOTA图像预处理 - 切片处理"""
        # 实现图像切片策略以适应大尺寸图像
        tiles = self.slice_image(image, tile_size=1024, overlap=200)
        return tiles
    
    def merge_dota_predictions(self, tile_predictions):
        """合并切片预测结果"""
        # 实现NMS和结果合并
        merged_boxes = self.nms_merge(tile_predictions)
        return merged_boxes