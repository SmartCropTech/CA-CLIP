import json
import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

class DiseaseSpotExtractor:
    def __init__(self, source_folder, disease_types=None):
            self.source_folder = source_folder
            self.disease_types = disease_types or ['Cedar-apple rust', 'Apple scab', 'Apple black rot', 
                'Common corn rust', 'Corn gray leaf spot',
                'Northern corn leaf blight', 'Grape isariopsis leaf spot',
                'Grape black rot', 'Grape black measles',
                'Potato early blight', 'Potato late blight',
                'Rice brown spot', 'Rice bacterial blight',
                'Rice blast', 'Tomato early blight',
                'Tomato late blight', 'Tomato septoria leaf spot',
                'Tomato stemphylium leaf spot', 'Wheat black rust', 'Wheat leaf blight',
                'Wheat yellow rust']
            self.disease_data = []
    
    def load_json(self, json_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def create_mask(self, shape, polygon_points):
        mask = np.zeros(shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [np.array(polygon_points, np.int32)], 255)
        return mask
    def create_lesion_mask(self, shape, polygon_points):
        mask = np.zeros(shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [np.array(polygon_points, np.int32)], 255)

        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return mask

        largest_contour = max(contours, key=cv2.contourArea)


        (x, y), radius = cv2.minEnclosingCircle(largest_contour)
        diameter = 2 * radius
        kernel_ratio=0.15

        kernel_size = max(1, int(diameter * kernel_ratio))

        kernel_size_odd = kernel_size * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size_odd, kernel_size_odd))

        dilated_mask = cv2.dilate(mask, kernel, iterations=1)

        return dilated_mask
    
    def calc_area(self, mask):
        return np.sum(mask == 255)

    def calc_avg_color(self, image, mask):
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        masked = cv2.bitwise_and(rgb, rgb, mask=mask)
        pixels = masked[mask > 0]
        return np.mean(pixels, axis=0).astype(int).tolist() if len(pixels) > 0 else [0,0,0]

    def calc_centroid(self, polygon_points):
        points = np.array(polygon_points, np.float32)
        M = cv2.moments(points)
        if M['m00'] != 0:
            cx, cy = M['m10']/M['m00'], M['m01']/M['m00']
        else:
            cx, cy = np.mean(points[:,0]), np.mean(points[:,1])
        return [float(cx), float(cy)]

    def calc_color_ratio(self, disease_color, leaf_color):
        return [round((disease_color[i]/leaf_color[i] if leaf_color[i] else 0), 3) for i in range(3)]

    def calc_distance_ratio(self, disease_centroid, leaf_centroid, leaf_mask):
        dc_dist = np.linalg.norm(np.array(disease_centroid) - np.array(leaf_centroid))
        direction = (np.array(disease_centroid) - np.array(leaf_centroid))
        if np.linalg.norm(direction) == 0:
            return 0, 0, 0
        direction = direction / np.linalg.norm(direction)
        h, w = leaf_mask.shape
        step = 0
        while True:
            x = int(leaf_centroid[0] + direction[0] * step)
            y = int(leaf_centroid[1] + direction[1] * step)
            if not (0 <= x < w and 0 <= y < h) or leaf_mask[y, x] == 0:
                break
            step += 1
        edge_dist = np.linalg.norm(direction * step)
        ratio = dc_dist / edge_dist if edge_dist > 0 else 0
        return round(ratio, 3), dc_dist, edge_dist
    
    def calc_angle(self, disease_centroid, leaf_centroid):
        dx = disease_centroid[0] - leaf_centroid[0]
        dy = disease_centroid[1] - leaf_centroid[1]
        

        angle_rad = math.atan2(dy, dx)
        angle_deg = math.degrees(angle_rad) % 360
        
        return round(angle_deg, 2)

    def extract_disease_region_image(self, image, mask):
        h, w = image.shape[:2]
        rgba_image = np.zeros((h, w, 4), dtype=np.uint8)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        rgba_image[:, :, :3] = rgb_image
        rgba_image[:, :, 3] = mask

        coords = cv2.findNonZero(mask)
        if coords is not None:
            x, y, w_rect, h_rect = cv2.boundingRect(coords)
            margin = 1  # 添加边距
            x = max(0, x - margin)
            y = max(0, y - margin)
            w_rect = min(w - x, w_rect + 2 * margin)
            h_rect = min(h - y, h_rect + 2 * margin)

            return rgba_image[y:y + h_rect, x:x + w_rect]

        return rgba_image
        
    def extract(self):
        for file in os.listdir(self.source_folder):
            if not file.endswith('.json'):
                continue

            json_path = os.path.join(self.source_folder, file)
            data = self.load_json(json_path)
            image_path = os.path.join(self.source_folder, data['imagePath'])
            img = cv2.imread(image_path)

            if img is None:
                print(f"Not image: {image_path}")
                continue

     
            leaf_mask = None
            leaf_centroid = None
            for s in data['shapes']:
                if s['label'] == 'Complete leaf':
                    leaf_mask = self.create_mask(img.shape, s['points'])
                    leaf_centroid = self.calc_centroid(s['points'])
                    break

            if leaf_mask is None:
                print(f"⚠ 未找到完整叶片标注: {file}")
                continue

            leaf_area = self.calc_area(leaf_mask)

         
            healthy_mask = leaf_mask.copy()

       
            disease_shapes = [s for s in data['shapes'] if s['label'] in self.disease_types]

          
            for disease_shape in disease_shapes:
                disease_mask = self.create_mask(img.shape, disease_shape['points'])
                healthy_mask = cv2.subtract(healthy_mask, disease_mask)

           
            leaf_color = self.calc_avg_color(img, healthy_mask)

          
            for disease_shape in disease_shapes:
                mask = self.create_mask(img.shape, disease_shape['points'])
                
                area = self.calc_area(mask)
                area_ratio = area / leaf_area
                dis_color = self.calc_avg_color(img, mask)
                color_ratio = self.calc_color_ratio(dis_color, leaf_color)
                dis_centroid = self.calc_centroid(disease_shape['points'])
                dist_ratio, d2c, c2e = self.calc_distance_ratio(dis_centroid, leaf_centroid, leaf_mask)
                
              
                angle = self.calc_angle(dis_centroid, leaf_centroid)

                lesion_mask = self.create_lesion_mask(img.shape, disease_shape['points'])
                
                disease_region = self.extract_disease_region_image(img, mask)
                disease_region_e = self.extract_disease_region_image(img, lesion_mask)

                self.disease_data.append({
                    "file": file,
                    "type": disease_shape['label'],
                    "area": area,
                    "leaf_area": leaf_area,
                    "area_ratio": round(area_ratio, 5),
                    "disease_color": dis_color,
                    "leaf_color": leaf_color,
                    "color_ratio": color_ratio,
                    "distance_ratio": dist_ratio,
                    "dist_d2c": round(d2c, 2),
                    "dist_c2e": round(c2e, 2),
                    "angle": angle,  
                    "disease_image": disease_region, 
                    "disease_region_e":disease_region_e
                })

        print(f"{len(self.disease_data)} lesions")
        return pd.DataFrame(self.disease_data)

    def get_disease_data(self):
        return self.disease_data
        
    def show_statistics(self):
        if not self.disease_data:
            print("暂无病斑数据。请先运行 extractor.extract()")
            return

        df = pd.DataFrame(self.disease_data)
        grouped = df.groupby('type')
    def show_healthy_example(self, index=0):
        if index >= len(self.disease_data):
            print(f"索引 {index} 超出范围，最大索引为 {len(self.disease_data)-1}")
            return
            
        spot = self.disease_data[index]
        disease_type = spot['type']
        file_name = spot['file']
        

        json_path = os.path.join(self.source_folder, file_name)
        data = self.load_json(json_path)
        image_path = os.path.join(self.source_folder, data['imagePath'])
        img = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        leaf_mask = np.zeros(img.shape[:2], dtype=np.uint8)
        disease_mask = np.zeros_like(leaf_mask)
        leaf_centroid = None
        disease_centroid = None

        for s in data['shapes']:
            if s['label'] == 'Complete leaf':
                cv2.fillPoly(leaf_mask, [np.array(s['points'], np.int32)], 255)
                leaf_centroid = self.calc_centroid(s['points'])
            if s['label'] == disease_type:
                cv2.fillPoly(disease_mask, [np.array(s['points'], np.int32)], 255)
                disease_centroid = self.calc_centroid(s['points'])

        healthy_mask = cv2.subtract(leaf_mask, disease_mask)
