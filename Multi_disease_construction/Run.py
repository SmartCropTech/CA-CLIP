
from Step_1_lesion_extract import DiseaseSpotExtractor 
from Step_2_image_construction import DiseaseAnalyzer, select_specific_lesions, process_single_image_with_specific_lesions,select_random_lesions_by_type

import json
import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import random


def batch_process_images(input_folder,counter,disease_data, output_folder,disease_types,filename_prefix,analyzer=None):
    if analyzer is None:
        analyzer = DiseaseAnalyzer()
    os.makedirs(output_folder, exist_ok=True)

    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    
    for image_file in image_files:
        image_path = os.path.join(input_folder, image_file)
        
        base_name = os.path.splitext(image_file)[0]
        json_path = os.path.join(input_folder, base_name + '.json')
        

        if not os.path.exists(json_path):
            continue
        type_counts = {}
        for disease_type in disease_types:
            if disease_type == "Apple black rot":
                rand_val = random.random()
                if rand_val < 0.3:  
                    lesion_count = 1
                elif 0.3 <= rand_val < 0.6:
                    lesion_count = 2
                elif 0.6 <= rand_val < 0.9:
                    lesion_count = 3
                else: 
                    lesion_count = 4
                type_counts[disease_type] = lesion_count

            elif disease_type == "Apple scab":
                rand_val = random.random()
                if rand_val < 0.1:  
                    lesion_count = 2
                elif 0.1 <= rand_val < 0.3:
                    lesion_count = 3
                elif 0.3 <= rand_val < 0.5:
                    lesion_count = 4
                elif 0.5 <= rand_val < 0.8:
                    lesion_count = 5
                else: 
                    lesion_count = 6
                type_counts[disease_type] = lesion_count

            elif disease_type == "Cedar-apple rust":
                rand_val = random.random()  
                if rand_val < 0.1: 
                    lesion_count = 1
                elif 0.2 <= rand_val < 0.5:
                    lesion_count = 2
                elif 0.5 <= rand_val < 0.7:
                    lesion_count = 3
                elif 0.7 <= rand_val < 0.8:
                    lesion_count = 4
                else:  
                    lesion_count = 5
                type_counts[disease_type] = lesion_count

            
        selected_lesions = select_random_lesions_by_type(disease_data, type_counts, seed=random.randint(1, 1000))
        
        output_filename = f"{filename_prefix}_{counter}.jpg"
        output_path = os.path.join(output_folder, output_filename)

        result_image = process_single_image_with_specific_lesions(
            image_path=image_path,
            json_path=json_path,
            disease_data=disease_data,
            selected_lesions=selected_lesions,
            output_path=output_path,
            analyzer=analyzer,
            show_plots=False
        )
        
        if result_image is not None:
            print(f"成功处理并保存: {output_filename}")
        else:
            print(f"处理失败: {image_file}")
        
        counter += 1

disease_types = ['Cedar-apple rust', 'Apple scab', 'Apple black rot', 
                'Common corn rust', 'Corn gray leaf spot',
                'Northern corn leaf blight', 'Grape isariopsis leaf spot',
                'Grape black rot', 'Grape black measles',
                'Potato early blight', 'Potato late blight',
                'Rice brown spot', 'Rice bacterial blight',
                'Rice blast', 'Tomato early blight',
                'Tomato late blight', 'Tomato septoria leaf spot',
                'Tomato stemphylium leaf spot', 'Wheat black rust', 'Wheat leaf blight',
                'Wheat yellow rust']

source_folder = "./Multi_disease_construction/Case_folder/source_folder"
extractor= DiseaseSpotExtractor(source_folder, disease_types)
# disease_data
disease_data = extractor.get_disease_data()
df = extractor.extract() 
# display(df.head(70))  

# 
input_folder = "./Multi_disease_construction/Case_folder/input_folder"
output_folder = "./Multi_disease_construction/Case_folder/output_folder"

# disease_types = ['Cedar-apple rust', 'Apple scab', 'Apple black rot', 
#                 'Common corn rust', 'Corn gray leaf spot',
#                 'Northern corn leaf blight', 'Grape isariopsis leaf spot',
#                 'Grape black rot', 'Grape black measles',
#                 'Potato early blight', 'Potato late blight',
#                 'Rice brown spot', 'Rice bacterial blight',
#                 'Rice blast', 'Tomato early blight',
#                 'Tomato late blight', 'Tomato septoria leaf spot',
#                 'Tomato stemphylium leaf spot', 'Wheat black rust', 'Wheat leaf blight',
#                 'Wheat yellow rust']

disease_types=["Apple black rot"]


output_name= ["Apple black rot",'Apple scab']       
filename_prefix = '+'.join(output_name)
analyzer = DiseaseAnalyzer()
counter=1
# 批量处理
batch_process_images(input_folder,counter,disease_data, output_folder,disease_types,filename_prefix,analyzer)