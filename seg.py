# we ae working on a human segementaiton project, where each image has one human or more 
# in it , and it captures part of the human body, like head, hand, foot, etc. 
# we need to segment the human in the image, and then save the segmented mask and the image 
# use the sam cpu for segmentation 

import cv2
import numpy as np
import torch
import os
import argparse
import urllib.request
from segment_anything import sam_model_registry, SamPredictor
from tqdm import tqdm
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection


def download_models():
    """
    下载所需的模型文件
    """
    # 下载 SAM 模型
    sam_model_urls = {
        "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
        "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
        "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
    }
    
    # 下载 SAM 模型
    sam_checkpoint = "sam_vit_h_4b8939.pth"
    if not os.path.exists(sam_checkpoint):
        print(f"正在下载SAM模型...")
        urllib.request.urlretrieve(sam_model_urls["vit_h"], sam_checkpoint)
        print(f"SAM模型下载完成")

def init_models(device="cpu"):
    """
    初始化模型
    
    Args:
        device: 运行设备
    
    Returns:
        sam_predictor: SAM预测器
        grounding_dino_model: Grounding DINO模型
        processor: Grounding DINO处理器
    """
    # 下载模型
    download_models()
    
    # 初始化SAM
    print("正在加载SAM模型...")
    sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h_4b8939.pth")
    sam.to(device=device)
    sam_predictor = SamPredictor(sam)
    
    # 初始化Grounding DINO
    print("正在加载Grounding DINO模型...")
    model_id = "IDEA-Research/grounding-dino-tiny"
    
    # 检查本地模型缓存目录
    import os
    from huggingface_hub import snapshot_download
    
    # 获取本地缓存路径
    cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
    model_cache_path = os.path.join(cache_dir, "models--" + model_id.replace("/", "--"))
    
    if os.path.exists(model_cache_path):
        print(f"发现本地模型缓存: {model_cache_path}")
        print("从本地加载Grounding DINO模型...")
        processor = AutoProcessor.from_pretrained(model_id, local_files_only=True)
        model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id, local_files_only=True).to(device)
    else:
        print(f"本地未找到模型缓存，从网络下载: {model_id}")
        print("正在下载Grounding DINO模型...")
        processor = AutoProcessor.from_pretrained(model_id)
        model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
    
    print("Grounding DINO模型加载成功")
    
    return sam_predictor, model, processor

def process_image_for_mask(sam_predictor, grounding_dino_model, processor, image, device="cpu"):
    """
    处理单张图像并返回掩码（不保存文件）
    
    Args:
        sam_predictor: SAM预测器
        grounding_dino_model: Grounding DINO模型
        processor: Grounding DINO处理器
        image: 输入图像cv2    
        device: 运行设备
    
    Returns:
        mask: 分割掩码，如果没有检测到目标则返回None
    """
    # 读取图像
    if image is None:
        print(f"错误：无法读取图像: ")
        return None
    
    # 转换为PIL图像
    image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    # 准备文本标签
    text_prompt = "fish crab "
    
    # 处理输入
    inputs = processor(images=image_pil, text=text_prompt, return_tensors="pt").to(device)
    
    # 进行检测
    with torch.no_grad():
        outputs = grounding_dino_model(**inputs)
    
    # 后处理检测结果
    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        text_threshold=0.25,
        target_sizes=[image_pil.size[::-1]]
    )
    
    result = results[0]
    
    # 打印详细的检测结果
    print("\n检测结果详情:")
    print(f"检测到的目标数量: {len(result['boxes'])}")
    
    # 如果没有检测到任何框，直接返回None，避免后续SAM预测报错
    if len(result["boxes"]) == 0:
        print("无检测结果，跳过SAM分割并返回None。")
        return None

    # 转换为RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 设置图像
    sam_predictor.set_image(image_rgb)
    
    # 使用检测框作为提示
    boxes = torch.tensor([box.tolist() for box in result["boxes"]], device=device)

    print(f"boxes: {boxes}")
    transformed_boxes = sam_predictor.transform.apply_boxes_torch(boxes, image_rgb.shape[:2])
    
    # 预测掩码
    masks, scores, logits = sam_predictor.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes,
        multimask_output=False
    )
    
    # 合并所有掩码
    combined_mask = torch.zeros_like(masks[0][0], dtype=torch.bool)
    for mask in masks:
        combined_mask = torch.logical_or(combined_mask, mask[0])
    
    # 转换为numpy数组
    combined_mask = combined_mask.cpu().numpy()
    
    return combined_mask

def process_image(sam_predictor, grounding_dino_model, processor, image_path, output_dir, device="cpu"):
    """
    处理单张图像
    
    Args:
        sam_predictor: SAM预测器
        grounding_dino_model: Grounding DINO模型
        processor: Grounding DINO处理器
        image_path: 输入图像路径
        output_dir: 输出目录
        device: 运行设备
    """
    # 读取图像
    image = cv2.imread(image_path)
    process_image_cv2(sam_predictor, grounding_dino_model, processor, image, image_path, output_dir, device)



def process_image_cv2(sam_predictor, grounding_dino_model, processor, image, output_dir, device="cpu"):
    if image is None:
        print(f"错误：无法读取图像: ")
        return
    
    # 转换为PIL图像
    image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    print("尝试检测人体...")
    
    # 准备文本标签 - 修改为单个字符串
    text_prompt = "fish. crab. marine animal"
    
    # 处理输入
    inputs = processor(images=image_pil, text=text_prompt, return_tensors="pt").to(device)
    
    # 进行检测
    with torch.no_grad():
        outputs = grounding_dino_model(**inputs)
    
    # 后处理检测结果
    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        text_threshold=0.25,
        target_sizes=[image_pil.size[::-1]]
    )
    
    # 获取第一个图像的结果
    result = results[0]
    
    # 打印详细的检测结果
    print("\n检测结果详情:")
    print(f"检测到的目标数量: {len(result['boxes'])}")
    for i, (box, score, label) in enumerate(zip(result["boxes"], result["scores"], result["labels"])):
        print(f"目标 {i+1}:")
        print(f"  标签: {label}")
        print(f"  置信度: {score.item():.3f}")
        print(f"  边界框: {box.tolist()}")
    
    if len(result["boxes"]) > 0:
        print(f"\n成功检测到 {len(result['boxes'])} 个人体目标")
        
        # 创建检测结果可视化
        dino_vis = image.copy()
        
        # 保存原始检测框
        for box, score, label in zip(result["boxes"], result["scores"], result["labels"]):
            # 转换坐标为整数
            x1, y1, x2, y2 = [int(coord) for coord in box.tolist()]
            
            print(f"\n绘制边界框:")
            print(f"原始坐标: x1={x1}, y1={y1}, x2={x2}, y2={y2}")
            
            # 确保坐标在图像范围内
            x1 = max(0, min(x1, image.shape[1]-1))
            y1 = max(0, min(y1, image.shape[0]-1))
            x2 = max(0, min(x2, image.shape[1]-1))
            y2 = max(0, min(y2, image.shape[0]-1))
            
            print(f"调整后坐标: x1={x1}, y1={y1}, x2={x2}, y2={y2}")
            print(f"图像尺寸: width={image.shape[1]}, height={image.shape[0]}")
            
            # 绘制边界框
            cv2.rectangle(dino_vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # 准备标签文本
            score_value = score.item()
            label_text = f"Score: {score_value:.2f}"
            
            # 获取文本大小
            (text_width, text_height), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            
            # 绘制标签背景
            cv2.rectangle(dino_vis, (x1, y1-text_height-10), (x1+text_width, y1), (0, 255, 0), -1)
            
            # 绘制标签文本
            cv2.putText(dino_vis, label_text, (x1, y1-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)  # 黑色文本
            
            # 添加标签名称
            label_name = f"Label: {label}"
            (label_width, label_height), _ = cv2.getTextSize(label_name, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            
            # 绘制标签名称背景
            cv2.rectangle(dino_vis, (x1, y2), (x1+label_width, y2+label_height+10), (0, 255, 0), -1)
            
            # 绘制标签名称文本
            cv2.putText(dino_vis, label_name, (x1, y2+label_height+5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)  # 黑色文本
        
        # 保存检测结果
        # base_name = os.path.splitext(os.path.basename(image_path))[0]
        # dino_path = os.path.join(output_dir, f"{base_name}_dino_detection.png")
        # cv2.imwrite(dino_path, dino_vis)
        # print(f"已保存Grounding DINO检测结果: {dino_path}")
        #return dino_vis
        
        # 保存裁剪出的人体区域
        for i, (box, score, label) in enumerate(zip(result["boxes"], result["scores"], result["labels"])):
            x1, y1, x2, y2 = [int(coord) for coord in box.tolist()]
            # 确保坐标在图像范围内
            x1 = max(0, min(x1, image.shape[1]-1))
            y1 = max(0, min(y1, image.shape[0]-1))
            x2 = max(0, min(x2, image.shape[1]-1))
            y2 = max(0, min(y2, image.shape[0]-1))
            
            # 裁剪人体区域
            person_crop = image[y1:y2, x1:x2]
            #if person_crop.size > 0:  # 确保裁剪区域有效
                # crop_path = os.path.join(output_dir, f"{base_name}_person_{i}.png")
                # cv2.imwrite(crop_path, person_crop)
                # print(f"已保存人体区域 {i}: {crop_path}")
        
        # 转换为RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 设置图像
        sam_predictor.set_image(image_rgb)
        
        # 使用检测框作为提示
        boxes = torch.tensor([box.tolist() for box in result["boxes"]], device=device)
        transformed_boxes = sam_predictor.transform.apply_boxes_torch(boxes, image_rgb.shape[:2])
        
        # 预测掩码
        masks, scores, logits = sam_predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=False
        )
        
        # 合并所有掩码
        combined_mask = torch.zeros_like(masks[0][0], dtype=torch.bool)
        for mask in masks:
            combined_mask = torch.logical_or(combined_mask, mask[0])
        
        # 转换为numpy数组
        combined_mask = combined_mask.cpu().numpy()
        print(f"combined_mask: {combined_mask}")
        #return combined_mask

        # 创建输出文件名
        mask_path = os.path.join(output_dir, f"{base_name}_mask.png")
        
        # 保存掩码
        cv2.imwrite(mask_path, combined_mask.astype(np.uint8) * 255)
        print(f"已保存掩码: {mask_path}")
        
        # 创建可视化结果
        colored_mask = np.zeros_like(image)
        colored_mask[combined_mask] = [0, 255, 0]  # 绿色掩码
        alpha = 0.5
        visualization = cv2.addWeighted(image, 1, colored_mask, alpha, 0)
        vis_path = os.path.join(output_dir, f"{base_name}_vis.png")
        cv2.imwrite(vis_path, visualization)
        print(f"已保存可视化结果: {vis_path}")
        
        # 保存带掩码的裁剪区域
        for i, box in enumerate(result["boxes"]):
            x1, y1, x2, y2 = [int(coord) for coord in box.tolist()]
            # 确保坐标在图像范围内
            x1 = max(0, min(x1, image.shape[1]-1))
            y1 = max(0, min(y1, image.shape[0]-1))
            x2 = max(0, min(x2, image.shape[1]-1))
            y2 = max(0, min(y2, image.shape[0]-1))
            
            # 裁剪带掩码的区域
            mask_crop = combined_mask[y1:y2, x1:x2]
            if mask_crop.size > 0:  # 确保裁剪区域有效
                mask_crop_path = os.path.join(output_dir, f"{base_name}_mask_crop_{i}.png")
                cv2.imwrite(mask_crop_path, mask_crop.astype(np.uint8) * 255)
                print(f"已保存带掩码的裁剪区域 {i}: {mask_crop_path}")
    else:
        # print(f"\n警告：在图像 {image_path} 中未检测到人体")
        print("可能的原因：")
        print("1. 图像中确实没有人")
        print("2. 人体姿态不常见")
        print("3. 图像质量不佳")
        print("4. 人体被遮挡")
        print("\n建议：")
        print("1. 检查图像是否包含人体")
        print("2. 尝试调整检测阈值")
        print("3. 使用不同的提示词")
        return

def main():
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='使用SAM和Grounding DINO进行人体分割')
    parser.add_argument('--input_dir', type=str, required=True,
                      help='输入图像目录')
    parser.add_argument('--output_dir', type=str, required=True,
                      help='输出目录')
    parser.add_argument('--device', type=str, default='cuda',
                      choices=['cpu', 'cuda'],
                      help='运行设备')
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 初始化模型
    sam_predictor, grounding_dino_model, processor = init_models(args.device)
    
    # 获取所有图像文件
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = []
    for ext in image_extensions:
        image_files.extend([f for f in os.listdir(args.input_dir) if f.lower().endswith(ext)])
    
    # 处理每张图像
    print(f"开始处理 {len(image_files)} 张图像...")
    for image_file in tqdm(image_files):
        image_path = os.path.join(args.input_dir, image_file)
        process_image(sam_predictor, grounding_dino_model, processor, image_path, args.output_dir, args.device)
    
    print(f"处理完成！结果保存在: {args.output_dir}")

if __name__ == "__main__":
    main()

