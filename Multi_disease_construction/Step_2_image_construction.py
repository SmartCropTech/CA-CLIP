import os
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import random
from PIL import Image
from scipy import ndimage

class DiseaseAnalyzer:
    def __init__(self):
        self.disease_types = ['Cedar-apple rust', 'Apple scab', 'Apple black rot', 
                                'Common corn rust', 'Corn gray leaf spot',
                                'Northern corn leaf blight', 'Grape isariopsis leaf spot',
                                'Grape black rot', 'Grape black measles',
                                'Potato early blight', 'Potato late blight',
                                'Rice brown spot', 'Rice bacterial blight',
                                'Rice blast', 'Tomato early blight',
                                'Tomato late blight', 'Tomato septoria leaf spot',
                                'Tomato stemphylium leaf spot', 'Wheat black rust', 'Wheat leaf blight',
                                'Wheat yellow rust']
    
    def create_mask(self, shape, polygon_points):
        mask = np.zeros(shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [np.array(polygon_points, np.int32)], 255)
        return mask
    
    def calc_area(self, mask):
        return np.sum(mask == 255)

    def calc_avg_color(self, image, mask):
        """计算掩码区域平均颜色 (RGB)"""
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
    
    def find_suitable_position_with_angle(self, leaf_mask, leaf_centroid, source_distance_ratio, source_angle, lesion_image=None):
        """
        在目标叶片上找到合适的位置，保持与源图像相同的角度和距离比例
        考虑病斑大小以避免超出边界
        """
        h, w = leaf_mask.shape

        # 直接使用源角度
        selected_angle = source_angle

        # 将角度转换为弧度
        rad = math.radians(selected_angle)

        # 计算方向向量
        direction = [math.cos(rad), math.sin(rad)]

        # 如果提供了病斑图像，考虑其大小
        lesion_half_width = 0
        lesion_half_height = 0
        if lesion_image is not None:
            lesion_half_width = lesion_image.shape[1] // 2
            lesion_half_height = lesion_image.shape[0] // 2

        # 沿着选定方向找到最大距离（考虑病斑大小）
        max_distance = 0
        step = 0
        max_steps = max(w, h)

        while step < max_steps:
            # 计算病斑中心位置
            center_x = int(leaf_centroid[0] + direction[0] * step)
            center_y = int(leaf_centroid[1] + direction[1] * step)
            
            # 计算病斑边界
            x_start = center_x - lesion_half_width
            y_start = center_y - lesion_half_height
            x_end = center_x + lesion_half_width
            y_end = center_y + lesion_half_height
            
            # 检查整个病斑区域是否在叶片内
            if (x_start >= 0 and y_start >= 0 and x_end < w and y_end < h and
                np.all(leaf_mask[y_start:y_end, x_start:x_end] > 0)):
                step += 1
            else:
                break

        max_distance = step

        # 计算目标距离
        target_distance = source_distance_ratio * max_distance

        # 如果找不到足够距离，尝试减小距离比例
        if max_distance == 0:
            # 尝试减小距离比例
            reduced_ratio = source_distance_ratio * 0.5

            # 重新计算最大距离
            step = 1
            while step < max_steps:
                center_x = int(leaf_centroid[0] + direction[0] * step)
                center_y = int(leaf_centroid[1] + direction[1] * step)
                
                x_start = center_x - lesion_half_width
                y_start = center_y - lesion_half_height
                x_end = center_x + lesion_half_width
                y_end = center_y + lesion_half_height
                
                if (x_start >= 0 and y_start >= 0 and x_end < w and y_end < h and
                    np.all(leaf_mask[y_start:y_end, x_start:x_end] > 0)):
                    step += 1
                else:
                    break

            max_distance = step
            target_distance = reduced_ratio * max_distance

        # 如果还是找不到，尝试在角度上做小范围调整
        if max_distance == 0:
            angle_adjustments = [0, 5, -5, 10, -10, 15, -15]
            for adjustment in angle_adjustments:
                adjusted_angle = (selected_angle + adjustment) % 360
                rad = math.radians(adjusted_angle)
                direction = [math.cos(rad), math.sin(rad)]

                step = 0
                while step < max_steps:
                    center_x = int(leaf_centroid[0] + direction[0] * step)
                    center_y = int(leaf_centroid[1] + direction[1] * step)
                    
                    x_start = center_x - lesion_half_width
                    y_start = center_y - lesion_half_height
                    x_end = center_x + lesion_half_width
                    y_end = center_y + lesion_half_height
                    
                    if (x_start >= 0 and y_start >= 0 and x_end < w and y_end < h and
                        np.all(leaf_mask[y_start:y_end, x_start:x_end] > 0)):
                        step += 1
                    else:
                        break

                if step > 0:
                    max_distance = step
                    target_distance = source_distance_ratio * max_distance
                    selected_angle = adjusted_angle
                    break

        # 如果仍然找不到，返回None
        if max_distance == 0 or max_distance >= max_steps:
            return None, selected_angle

        # 计算目标位置
        target_x = int(leaf_centroid[0] + direction[0] * target_distance)
        target_y = int(leaf_centroid[1] + direction[1] * target_distance)

        # 确保整个病斑都在叶片内
        x_start = target_x - lesion_half_width
        y_start = target_y - lesion_half_height
        x_end = target_x + lesion_half_width
        y_end = target_y + lesion_half_height
        
        if (x_start >= 0 and y_start >= 0 and x_end < w and y_end < h and
            np.all(leaf_mask[y_start:y_end, x_start:x_end] > 0)):
            return (target_x, target_y), selected_angle

        return None, selected_angle

    def find_random_position_within_leaf(self, leaf_mask, lesion_image, existing_disease_masks, max_attempts=50):
        """
        在叶片内随机寻找不重叠的位置
        """
        h, w = leaf_mask.shape
        h_lesion, w_lesion = lesion_image.shape[:2]
        
        for attempt in range(max_attempts):
            # 随机生成位置
            x = random.randint(w_lesion // 2, w - w_lesion // 2 - 1)
            y = random.randint(h_lesion // 2, h - h_lesion // 2 - 1)
            
            # 检查是否在叶片内
            x_start = x - w_lesion // 2
            y_start = y - h_lesion // 2
            x_end = x_start + w_lesion
            y_end = y_start + h_lesion
            
            if (x_start >= 0 and y_start >= 0 and x_end < w and y_end < h and
                np.all(leaf_mask[y_start:y_end, x_start:x_end] > 0)):
                
                # 检查是否重叠
                if not self.check_overlap((x, y), lesion_image, existing_disease_masks, leaf_mask):
                    return (x, y)
        
        return None

    def check_overlap(self, position, lesion_image, existing_disease_masks, leaf_mask=None):
        """
        检查病斑是否与现有病斑重叠
        """
        # 获取病斑尺寸
        h_lesion, w_lesion = lesion_image.shape[:2]
        center_x, center_y = position
        
        # 计算粘贴区域
        x_start = center_x - w_lesion // 2
        y_start = center_y - h_lesion // 2
        x_end = x_start + w_lesion
        y_end = y_start + h_lesion
        
        # 确保不超出图像边界
        img_height, img_width = existing_disease_masks[0].shape if existing_disease_masks else (leaf_mask.shape if leaf_mask is not None else (0, 0))
        if x_start < 0 or y_start < 0 or x_end > img_width or y_end > img_height:
            return True
            
        # 检查是否在叶片外（如果提供了叶片掩码）
        if leaf_mask is not None:
            lesion_alpha = lesion_image[:, :, 3]
            for y in range(max(0, y_start), min(img_height, y_end)):
                for x in range(max(0, x_start), min(img_width, x_end)):
                    if (lesion_alpha[y-y_start, x-x_start] > 0 and 
                        (y >= leaf_mask.shape[0] or x >= leaf_mask.shape[1] or leaf_mask[y, x] == 0)):
                        return True
        
        # 提取病斑的Alpha通道作为掩码
        lesion_alpha = lesion_image[:, :, 3]
        
        # 创建一个与现有病斑掩码相同大小的临时掩码
        temp_mask = np.zeros_like(existing_disease_masks[0]) if existing_disease_masks else np.zeros_like(leaf_mask)
        
        # 将病斑掩码放置到临时掩码中
        roi_y_start = max(0, y_start)
        roi_y_end = min(temp_mask.shape[0], y_end)
        roi_x_start = max(0, x_start)
        roi_x_end = min(temp_mask.shape[1], x_end)
        
        lesion_y_start = max(0, -y_start)
        lesion_y_end = lesion_y_start + (roi_y_end - roi_y_start)
        lesion_x_start = max(0, -x_start)
        lesion_x_end = lesion_x_start + (roi_x_end - roi_x_start)
        
        if (roi_y_start < roi_y_end and roi_x_start < roi_x_end and
            lesion_y_start < lesion_y_end and lesion_x_start < lesion_x_end):
            roi = temp_mask[roi_y_start:roi_y_end, roi_x_start:roi_x_end]
            lesion_roi = lesion_alpha[lesion_y_start:lesion_y_end, lesion_x_start:lesion_x_end]
            roi[lesion_roi > 0] = 255
        
        # 检查每个现有病斑掩码
        for existing_mask in existing_disease_masks:
            # 检查是否有重叠
            overlap = np.any(np.logical_and(temp_mask > 0, existing_mask > 0))
            if overlap:
                return True
        
        return False
    
    def adjust_color(self, lesion_image, source_leaf_color, target_leaf_color):
        # 提取RGB和Alpha通道
        lesion_rgb = lesion_image[:, :, :3].astype(np.float32)
        lesion_alpha = lesion_image[:, :, 3]
        
        # 计算颜色调整比例 (避免除零)
        color_ratios = []
        for i in range(3):
            if source_leaf_color[i] > 0:
                ratio = target_leaf_color[i] / source_leaf_color[i]
            else:
                ratio = 1.0
            color_ratios.append(ratio)
        
        # 应用颜色调整
        for i in range(3):
            lesion_rgb[:, :, i] = lesion_rgb[:, :, i] * color_ratios[i]
        
        # 确保值在有效范围内
        lesion_rgb = np.clip(lesion_rgb, 0, 255).astype(np.uint8)
        
        # 重新组合图像
        adjusted_lesion = np.zeros_like(lesion_image)
        adjusted_lesion[:, :, :3] = lesion_rgb
        adjusted_lesion[:, :, 3] = lesion_alpha
        
        return adjusted_lesion
    
    def adjust_size(self, lesion_image, source_area_ratio, target_leaf_area, scale_factor=1.0):
        """
        根据源病斑面积比例和目标叶片面积调整病斑大小
        """
        # 计算目标病斑面积
        target_lesion_area = source_area_ratio * target_leaf_area * scale_factor
        
        # 计算当前病斑面积 (非透明像素数量)
        current_lesion_area = np.sum(lesion_image[:, :, 3] > 0)
        
        # 计算缩放比例
        if current_lesion_area > 0:
            scale = math.sqrt(target_lesion_area / current_lesion_area)
        else:
            scale = 1.0
        
        # 限制缩放范围，避免病斑过大或过小
        scale = max(0.3, min(2.0, scale))
        
        # 计算新尺寸
        h, w = lesion_image.shape[:2]
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # 调整图像大小
        if new_w > 0 and new_h > 0:
            # 使用PIL进行高质量缩放，保持透明度
            lesion_pil = Image.fromarray(lesion_image)
            resized_lesion = lesion_pil.resize((new_w, new_h), Image.LANCZOS)
            return np.array(resized_lesion)
        else:
            return lesion_image
    
    def create_polygon_mask(self, overlay_image):
        """
        从病斑图像创建多边形掩模
        """
        # 确保是numpy数组
        if isinstance(overlay_image, np.ndarray):
            overlay_array = overlay_image
        else:
            overlay_array = np.array(overlay_image)
        
        # 使用alpha通道作为掩模
        if overlay_array.shape[2] == 4:  # RGBA
            alpha_mask = overlay_array[:, :, 3] > 0
        else:  # RGB或其他
            # 如果没有alpha通道，创建一个全白的掩模
            alpha_mask = np.ones(overlay_array.shape[:2], dtype=bool)
        
        return alpha_mask
    
    def calculate_min_enclosing_circle_diameter(self, mask):
        """
        计算病斑最小外接圆的直径
        """
        # 找到病斑轮廓
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return 0
        
        # 找到最大的轮廓
        largest_contour = max(contours, key=cv2.contourArea)
        
        # 计算最小外接圆
        (x, y), radius = cv2.minEnclosingCircle(largest_contour)
        diameter = 2 * radius
        
        return diameter
    
    def calculate_feather_radius_by_diameter_percentage(self, mask, feather_percent=10):
        """
        根据病斑最小外接圆直径的百分比计算羽化半径
        """
        # 计算病斑最小外接圆直径
        diameter = self.calculate_min_enclosing_circle_diameter(mask)
        
        if diameter == 0:
            return 1
        
        # 计算羽化半径
        feather_radius = int(diameter * feather_percent / 100)
        
        # 确保羽化半径至少为1像素
        feather_radius = max(2, feather_radius)
        
        return feather_radius
    
    def create_feather_mask_for_polygon(self, polygon_mask, feather_radius):
        """
        为病斑多边形创建羽化掩模
        """
        # 计算到病斑边界的距离
        distance = ndimage.distance_transform_edt(polygon_mask)
        
        # 计算最大距离（病斑内部最远点的距离）
        max_distance = np.max(distance)
        
        # 创建羽化掩模
        feather_mask = np.zeros_like(distance, dtype=np.float32)
        
        # 在病斑内部区域应用羽化
        for i in range(polygon_mask.shape[0]):
            for j in range(polygon_mask.shape[1]):
                if polygon_mask[i, j]:
                    # 计算当前点到边界的距离
                    dist_to_edge = distance[i, j]
                    
                    # 如果距离小于羽化半径，则应用渐变
                    if dist_to_edge < feather_radius:
                        feather_mask[i, j] = dist_to_edge / feather_radius
                    else:
                        # 在病斑内部区域，完全混合
                        feather_mask[i, j] = 1.0
        
        return feather_mask
    
    def apply_feathering_to_lesion(self, lesion_image, feather_percent=10):
        """
        对病斑图像应用羽化效果
        """
        # 获取病斑掩模
        polygon_mask = self.create_polygon_mask(lesion_image)
        
        # 根据病斑最小外接圆直径百分比计算羽化半径
        feather_radius = self.calculate_feather_radius_by_diameter_percentage(polygon_mask, feather_percent)
        
        # 创建羽化掩模
        feather_mask = self.create_feather_mask_for_polygon(polygon_mask, feather_radius)

        # 将羽化掩模应用到病斑的alpha通道
        lesion_image_float = lesion_image.astype(np.float32) / 255.0
        lesion_image_float[:, :, 3] *= feather_mask
        
        # 转换回uint8
        feathered_lesion = (lesion_image_float * 255).astype(np.uint8)
        
        return feathered_lesion
    
    def paste_lesion_within_leaf(self, target_image, target_mask, position, lesion_image, disease_region_e,
                                source_leaf_color, target_leaf_color, source_area_ratio, target_leaf_area,
                                apply_feathering=True, feather_percent=10):
        """
        使用PIL将病斑粘贴到目标图像的指定位置，确保不超出叶片区域
        """
        # 调整病斑颜色
        color_adjusted_lesion = self.adjust_color(
            lesion_image, source_leaf_color, target_leaf_color
        )
        color_disease_region_e = self.adjust_color(
            disease_region_e, source_leaf_color, target_leaf_color
        )

        # 应用羽化效果（如果启用）
        if apply_feathering:
            final_lesion = self.apply_feathering_to_lesion(
                color_disease_region_e, feather_percent
            )
        else:
            final_lesion = color_disease_region_e
            
        color_adjusted_lesion = self.combine_lesion_images(color_adjusted_lesion, color_disease_region_e)
        
        # 将OpenCV图像转换为PIL图像
        target_pil = Image.fromarray(cv2.cvtColor(target_image, cv2.COLOR_BGR2RGB))
        lesion_pil = Image.fromarray(final_lesion)
        
        # 确保病斑图像是RGBA模式
        if lesion_pil.mode != 'RGBA':
            lesion_pil = lesion_pil.convert('RGBA')
        
        # 确保目标图像是RGBA模式
        if target_pil.mode != 'RGBA':
            target_pil = target_pil.convert('RGBA')
        
        # 计算粘贴位置（左上角坐标）
        x, y = position
        lesion_width, lesion_height = lesion_pil.size
        
        # 调整位置使病斑中心对准指定位置
        x = x - lesion_width // 2
        y = y - lesion_height // 2
        
        # 创建一个与目标图像相同大小的透明图层
        temp = Image.new("RGBA", target_pil.size, (0, 0, 0, 0))
        
        # 将病斑图像粘贴到透明图层的指定位置
        temp.paste(lesion_pil, (x, y), lesion_pil)
        
        # 将透明图层与目标图像合并
        result_pil = Image.alpha_composite(target_pil, temp)
        
        # 将结果转换回OpenCV格式
        result_rgba = np.array(result_pil)
        result_bgr = cv2.cvtColor(result_rgba, cv2.COLOR_RGBA2BGR)
        
        # 确保病斑不超出叶片区域
        leaf_only_mask = np.zeros_like(target_mask)
        leaf_only_mask[target_mask > 0] = 255
        
        # 将超出叶片区域的部分恢复为原图
        outside_leaf = leaf_only_mask == 0
        result_bgr[outside_leaf] = target_image[outside_leaf]
        
        return result_bgr
    
    def find_non_overlapping_positions(self, leaf_mask, leaf_centroid, selected_lesions, existing_disease_masks, target_leaf_area, max_attempts_per_lesion=20):
        """
        为选定的病斑找到不重叠的位置
        先调整大小，然后基于调整后的大小查找位置
        如果找不到位置，尝试缩小病斑面积并随机放置
        """
        positions = []
        temp_disease_masks = existing_disease_masks.copy()
        success_count = 0
        
        for i, lesion_data in enumerate(selected_lesions):
            placed = False
            original_lesion_image = lesion_data['disease_image']
            
            # 第一步：先调整病斑大小
            adjusted_lesion = self.adjust_size(
                original_lesion_image, 
                lesion_data['area_ratio'], 
                target_leaf_area
            )
            
            # 尝试在原始大小下找到位置
            for attempt in range(max_attempts_per_lesion):
                # 基于调整后的大小查找位置
                target_position, selected_angle = self.find_suitable_position_with_angle(
                    leaf_mask, leaf_centroid, 
                    lesion_data['distance_ratio'], 
                    lesion_data['angle'],
                    adjusted_lesion
                )
                
                # 如果找不到合适位置，尝试缩小病斑
                if target_position is None:
                    print(f"无法为病斑 {i+1} 找到合适位置，尝试缩小病斑")
                    break
                
                # 检查是否重叠
                if not self.check_overlap(target_position, adjusted_lesion, temp_disease_masks, leaf_mask):
                    positions.append((target_position, adjusted_lesion, lesion_data))
                    
                    # 创建新病斑的临时掩码并添加到列表中
                    new_mask = self.create_lesion_mask(adjusted_lesion, target_position, leaf_mask.shape)
                    temp_disease_masks.append(new_mask)
                    success_count += 1
                    placed = True
                    print(f"成功放置病斑 {i+1}/{len(selected_lesions)} (原始大小)")
                    break
            
            # 如果原始大小找不到位置，尝试缩小病斑
            if not placed:
                print(f"病斑 {i+1} 原始大小无法放置，尝试缩小病斑")
                
                # 尝试不同的缩小比例
                scale_factors = [0.8, 0.7, 0.6, 0.5,0.4,0.3,0.2]  # 逐步缩小到50%
                
                for scale_factor in scale_factors:
                    # 缩小病斑
                    shrunk_lesion = self.adjust_size(
                        original_lesion_image, 
                        lesion_data['area_ratio'], 
                        target_leaf_area,
                        scale_factor=scale_factor
                    )
                    
                    # 尝试随机放置缩小后的病斑
                    random_position = self.find_random_position_within_leaf(
                        leaf_mask, shrunk_lesion, temp_disease_masks, max_attempts=30
                    )
                    
                    if random_position is not None:
                        positions.append((random_position, shrunk_lesion, lesion_data))
                        
                        # 创建新病斑的临时掩码并添加到列表中
                        new_mask = self.create_lesion_mask(shrunk_lesion, random_position, leaf_mask.shape)
                        temp_disease_masks.append(new_mask)
                        success_count += 1
                        placed = True
                        print(f"成功放置病斑 {i+1}/{len(selected_lesions)} (缩小到{scale_factor*100}%)")
                        break
                    else:
                        print(f"病斑 {i+1} 缩小到{scale_factor*100}%仍无法放置")
            
            if not placed:
                positions.append((None, None, lesion_data))
                print(f"无法放置病斑 {i+1}/{len(selected_lesions)} (即使缩小到50%)")
        
        return positions, success_count
    
    def create_lesion_mask(self, lesion_image, position, mask_shape):
        """
        创建病斑的掩码
        """
        mask = np.zeros(mask_shape, dtype=np.uint8)
        h_lesion, w_lesion = lesion_image.shape[:2]
        x, y = position
        
        x_start = max(0, x - w_lesion // 2)
        y_start = max(0, y - h_lesion // 2)
        x_end = min(mask_shape[1], x_start + w_lesion)
        y_end = min(mask_shape[0], y_start + h_lesion)
        
        lesion_alpha = lesion_image[:, :, 3]
        lesion_y_start = max(0, - (y - h_lesion // 2))
        lesion_y_end = lesion_y_start + (y_end - y_start)
        lesion_x_start = max(0, - (x - w_lesion // 2))
        lesion_x_end = lesion_x_start + (x_end - x_start)
        
        if (lesion_y_start < lesion_y_end and lesion_x_start < lesion_x_end and
            y_start < y_end and x_start < x_end):
            lesion_roi = lesion_alpha[lesion_y_start:lesion_y_end, lesion_x_start:lesion_x_end]
            mask[y_start:y_end, x_start:x_end] = (lesion_roi > 0) * 255
        
        return mask
    
    def combine_lesion_images(self, size_adjusted_lesion, lesion_image):            
        if size_adjusted_lesion.shape != lesion_image.shape:
            # 如果尺寸不同，将lesion_image调整为与size_adjusted_lesion相同的尺寸
            lesion_pil = Image.fromarray(lesion_image)
            lesion_pil = lesion_pil.resize((size_adjusted_lesion.shape[1], size_adjusted_lesion.shape[0]), Image.LANCZOS)
            lesion_image = np.array(lesion_pil)

        # 获取两个图像的二值掩模
        mask_adjusted = self.create_polygon_mask(size_adjusted_lesion)
        mask_original = self.create_polygon_mask(lesion_image)

        # 找出在原始图像中但不在调整后图像中的区域
        mask_only_in_original = np.logical_and(mask_original, ~mask_adjusted)

        # 创建结果图像，初始化为调整后的图像
        combined_lesion = size_adjusted_lesion.copy()

        # 将只在原始图像中的区域添加到结果图像中
        combined_lesion[mask_only_in_original] = lesion_image[mask_only_in_original]

        return combined_lesion


# 辅助函数
def select_specific_lesions(disease_data, lesion_indices):
    """
    根据索引选择特定的病斑
    
    参数:
        disease_data: 所有病斑数据列表
        lesion_indices: 要选择的病斑索引列表
        
    返回:
        selected_lesions: 选定的病斑列表
    """
    selected_lesions = []
    
    for idx in lesion_indices:
        if 0 <= idx < len(disease_data):
            selected_lesions.append(disease_data[idx])
            print(f"选择了病斑索引 {idx}: {disease_data[idx].get('type', 'Unknown')}")
        else:
            print(f"警告: 索引 {idx} 超出范围，跳过")
    
    print(f"总共选择了 {len(selected_lesions)} 个指定病斑")
    return selected_lesions


def select_lesions_by_type_and_index(disease_data, type_indices_dict):
    """
    按类型和索引选择病斑
    
    参数:
        disease_data: 所有病斑数据列表
        type_indices_dict: 字典，键为病害类型，值为该类型中要选择的索引列表
            例如: {'Cedar-apple rust': [0, 1], 'Apple scab': [2, 3]}
            
    返回:
        selected_lesions: 选定的病斑列表
    """
    selected_lesions = []
    
    # 按类型分组病斑
    type_groups = {}
    for lesion in disease_data:
        lesion_type = lesion.get('type', 'Unknown')
        if lesion_type not in type_groups:
            type_groups[lesion_type] = []
        type_groups[lesion_type].append(lesion)
    
    # 选择指定类型的指定索引
    for lesion_type, indices in type_indices_dict.items():
        if lesion_type in type_groups:
            type_lesions = type_groups[lesion_type]
            for idx in indices:
                if 0 <= idx < len(type_lesions):
                    selected_lesions.append(type_lesions[idx])
                    print(f"选择了 {lesion_type} 类型的病斑索引 {idx}")
                else:
                    print(f"警告: {lesion_type} 类型的索引 {idx} 超出范围，跳过")
        else:
            print(f"警告: 未找到 {lesion_type} 类型的病斑")
    
    print(f"总共选择了 {len(selected_lesions)} 个指定病斑")
    return selected_lesions


def select_random_lesions_by_type(disease_data, type_counts, seed=None):

    if seed is not None:
        random.seed(seed)
    
    selected_lesions = []
    
    # 按类型分组病斑
    type_groups = {}
    for lesion in disease_data:
        lesion_type = lesion.get('type', 'Unknown')
        if lesion_type in type_counts:  # 只关注指定的病害类型
            if lesion_type not in type_groups:
                type_groups[lesion_type] = []
            type_groups[lesion_type].append(lesion)
    
    # 从每种类型中随机挑选指定数量的病斑
    for lesion_type, count in type_counts.items():
        if lesion_type in type_groups:
            type_lesions = type_groups[lesion_type]
            if count > len(type_lesions):
                print(f"警告: {lesion_type} 类型只有 {len(type_lesions)} 个病斑，但要求选择 {count} 个")
                count = len(type_lesions)
            
            # 随机挑选指定数量的病斑
            selected = random.sample(type_lesions, count)
            selected_lesions.extend(selected)
            print(f"从 {lesion_type} 类型中随机选择了 {count} 个病斑")
            
            # 打印选中的病斑信息
            for i, lesion in enumerate(selected):
                print(f"  - {lesion_type} #{i+1}: 面积比例 {lesion.get('area_ratio', 'N/A')}, 距离比例 {lesion.get('distance_ratio', 'N/A')}, 角度 {lesion.get('angle', 'N/A')}")
        else:
            print(f"警告: 未找到 {lesion_type} 类型的病斑")
    
    print(f"总共选择了 {len(selected_lesions)} 个指定病斑")
    return selected_lesions


# 主处理函数 - 处理单个图像和JSON文件
def process_single_image_with_specific_lesions(image_path, json_path, disease_data, selected_lesions, output_path, 
                                              analyzer=None, show_plots=True):

    if analyzer is None:
        analyzer = DiseaseAnalyzer()
        
    if not selected_lesions:
        print("没有选择任何病斑进行粘贴")
        return
    
    print(f"将尝试粘贴 {len(selected_lesions)} 个指定病斑")
    
    try:
        # 读取JSON文件
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print("JSON路径:", json_path)
        print("图片路径:", image_path)
        
        # 读取图片
        img = cv2.imread(image_path)
        if img is None:
            print(f"无法读取图像: {image_path}")
            return
        
        # 获取完整叶片掩码
        leaf_mask = None
        leaf_centroid = None
        for s in data['shapes']:
            if s['label'] == 'Complete leaf':
                leaf_mask = analyzer.create_mask(img.shape, s['points'])
                leaf_centroid = analyzer.calc_centroid(s['points'])
                break

        if leaf_mask is None:
            print(f"⚠ 未找到完整叶片标注: {json_path}")
            return

        leaf_area = analyzer.calc_area(leaf_mask)

        # 创建健康区域掩码：先复制叶片掩码，然后减去所有病斑区域
        healthy_mask = leaf_mask.copy()

        # 收集所有病斑区域
        disease_shapes = [s for s in data['shapes'] if s['label'] in analyzer.disease_types]
        
        # 创建现有病斑掩码列表
        existing_disease_masks = []
        for disease_shape in disease_shapes:
            disease_mask = analyzer.create_mask(img.shape, disease_shape['points'])
            existing_disease_masks.append(disease_mask)
            healthy_mask = cv2.subtract(healthy_mask, disease_mask)

        # 计算健康区域的平均颜色
        leaf_color = analyzer.calc_avg_color(img, healthy_mask)
        
        print("叶片颜色:", leaf_color)
        print("叶片面积:", leaf_area)
        
        # 为所有病斑找到不重叠的位置
        positions, success_count = analyzer.find_non_overlapping_positions(
            leaf_mask, leaf_centroid, selected_lesions, existing_disease_masks, leaf_area
        )
        
        print(f"成功为 {success_count}/{len(selected_lesions)} 个病斑找到位置")
        
        # 创建结果图像
        result_img_no_feather = img.copy()
        result_img_feathered = img.copy()
        
        # 逐个粘贴病斑
        for position, adjusted_lesion, lesion_data in positions:
            if position is None:
                continue
            
            # 没有羽化的结果
            result_img_no_feather = analyzer.paste_lesion_within_leaf(
                result_img_no_feather, 
                leaf_mask, 
                position, 
                adjusted_lesion,
                lesion_data['disease_image'],
                lesion_data['leaf_color'],  # 源叶片颜色
                leaf_color,                 # 目标叶片颜色
                lesion_data['area_ratio'],  # 源病斑面积比例
                leaf_area,                  # 目标叶片面积
                apply_feathering=False      # 不应用羽化
            )
            
            # 有羽化的结果
            result_img_feathered = analyzer.paste_lesion_within_leaf(
                result_img_feathered, 
                leaf_mask, 
                position, 
                adjusted_lesion,
                lesion_data['disease_region_e'],
                lesion_data['leaf_color'],  # 源叶片颜色
                leaf_color,                 # 目标叶片颜色
                lesion_data['area_ratio'],  # 源病斑面积比例
                leaf_area,                  # 目标叶片面积
                apply_feathering=True,      # 应用羽化
                feather_percent=16          # 羽化百分比
            )
        
        # 保存结果图像
        cv2.imwrite(output_path, result_img_feathered)
        print(f"已保存有羽化结果: {output_path}")
        
        # 可视化结果（可选）
        if show_plots:
            plt.figure(figsize=(18, 6))
            
            # 显示原图
            plt.subplot(1, 3, 1)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            plt.imshow(img_rgb)
            plt.title('原始图像')
            plt.axis('off')
            
            # 显示没有羽化的结果
            plt.subplot(1, 3, 2)
            result_no_feather_rgb = cv2.cvtColor(result_img_no_feather, cv2.COLOR_BGR2RGB)
            plt.imshow(result_no_feather_rgb)
            plt.title(f'无羽化效果 ({success_count}个病斑)')
            plt.axis('off')
            
            # 显示有羽化的结果
            plt.subplot(1, 3, 3)
            result_feathered_rgb = cv2.cvtColor(result_img_feathered, cv2.COLOR_BGR2RGB)
            plt.imshow(result_feathered_rgb)
            plt.title(f'有羽化效果 ({success_count}个病斑)')
            plt.axis('off')
            
            plt.tight_layout()
            plt.show()
        
        return result_img_feathered
        
    except Exception as e:
        print(f"处理文件时发生错误: {e}")
        import traceback
        traceback.print_exc()
        return None

    