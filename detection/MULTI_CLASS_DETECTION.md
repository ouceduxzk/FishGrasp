# 多类别检测配置指南

## 🎯 类别配置

### 当前类别设置
```yaml
# datasets/l0_9.12/dataset.yaml
names: ['背景', '鱿鱼']
```

### 类别索引映射
- **类别 0**: 背景 (background)
- **类别 1**: 鱿鱼 (squid)

## 📊 训练输出解读

### 多类别检测的输出格式
```
Class     Images  Instances      Box(P          R      mAP50  mAP50-95)
all        241       1160      0.944      0.972      0.977      0.722
背景         241        800      0.950      0.980      0.985      0.750
鱿鱼         241        360      0.938      0.964      0.969      0.694
```

### 指标说明
- **all**: 所有类别的平均性能
- **背景**: 背景类别的检测性能
- **鱿鱼**: 鱿鱼类别的检测性能

## 🔧 训练命令

### 基本训练
```bash
python3 detection/train_yolo.py \
    --data ./datasets/l0_9.12/dataset.yaml \
    --model yolov8s.pt \
    --epochs 100 \
    --project runs/train \
    --name multi_class_squid_background_$(date +%Y%m%d_%H%M%S)
```

### 硬负样本挖掘训练
```bash
python3 detection/train_yolo_with_hard_negative.py \
    --data ./datasets/l0_9.12/dataset.yaml \
    --model yolov8s.pt \
    --epochs 100 \
    --project runs/train \
    --name multi_class_hard_negative_$(date +%Y%m%d_%H%M%S) \
    --mining_strategy confidence_based \
    --hard_negative_ratio 0.3
```

## 📈 性能分析

### 类别平衡性检查
```python
# 检查每个类别的样本数量
def analyze_class_distribution(dataset_path):
    train_labels = Path(dataset_path) / "labels" / "train"
    val_labels = Path(dataset_path) / "labels" / "val"
    
    class_counts = {0: 0, 1: 0}  # 背景, 鱿鱼
    
    for split in [train_labels, val_labels]:
        for label_file in split.glob("*.txt"):
            with open(label_file, 'r') as f:
                for line in f:
                    if line.strip():
                        class_id = int(line.split()[0])
                        class_counts[class_id] += 1
    
    print("类别分布:")
    print(f"背景 (类别0): {class_counts[0]} 个实例")
    print(f"鱿鱼 (类别1): {class_counts[1]} 个实例")
    
    return class_counts
```

### 类别不平衡处理
如果发现类别不平衡，可以：

1. **调整损失权重**
```python
# 在训练时使用类别权重
class_weights = [1.0, 2.0]  # 给鱿鱼更高权重
```

2. **数据增强**
```python
# 对少数类别进行更多增强
--aug strong
--mixup 0.3
--copy_paste 0.4
```

3. **硬负样本挖掘**
```python
# 重点关注困难样本
--mining_strategy confidence_based
--hard_negative_ratio 0.4
```

## 🎯 实际应用场景

### 场景1: 鱿鱼检测
- **目标**: 准确检测鱿鱼
- **策略**: 重点关注鱿鱼类别的性能
- **指标**: 鱿鱼的mAP50和Recall

### 场景2: 背景过滤
- **目标**: 减少背景误检
- **策略**: 提高背景类别的精确率
- **指标**: 背景的Precision

### 场景3: 平衡检测
- **目标**: 两个类别都表现良好
- **策略**: 关注整体mAP50
- **指标**: all类别的综合性能

## 📊 性能监控

### 训练过程监控
```python
# 监控每个类别的性能变化
def monitor_class_performance(results):
    for epoch, result in enumerate(results):
        print(f"Epoch {epoch}:")
        print(f"  背景 mAP50: {result['background_map50']:.3f}")
        print(f"  鱿鱼 mAP50: {result['squid_map50']:.3f}")
        print(f"  整体 mAP50: {result['overall_map50']:.3f}")
```

### 类别特定分析
```python
# 分析每个类别的困难样本
def analyze_class_specific_hard_negatives(hard_negatives):
    background_hard = [hn for hn in hard_negatives if hn['prediction']['class'] == 0]
    squid_hard = [hn for hn in hard_negatives if hn['prediction']['class'] == 1]
    
    print(f"背景困难样本: {len(background_hard)}")
    print(f"鱿鱼困难样本: {len(squid_hard)}")
```

## 🔍 调试技巧

### 1. 类别混淆分析
```python
# 分析类别间的混淆情况
def analyze_class_confusion(predictions, ground_truth):
    confusion_matrix = np.zeros((2, 2))  # 2x2矩阵
    
    for pred, gt in zip(predictions, ground_truth):
        pred_class = pred['class']
        gt_class = gt['class']
        confusion_matrix[gt_class][pred_class] += 1
    
    print("混淆矩阵:")
    print("        预测")
    print("实际    背景  鱿鱼")
    print(f"背景   {confusion_matrix[0][0]:.0f}   {confusion_matrix[0][1]:.0f}")
    print(f"鱿鱼   {confusion_matrix[1][0]:.0f}   {confusion_matrix[1][1]:.0f}")
```

### 2. 边界框质量分析
```python
# 分析每个类别的边界框质量
def analyze_bbox_quality(predictions, ground_truth):
    for class_id, class_name in enumerate(['背景', '鱿鱼']):
        class_predictions = [p for p in predictions if p['class'] == class_id]
        class_ground_truth = [g for g in ground_truth if g['class'] == class_id]
        
        ious = []
        for pred in class_predictions:
            max_iou = 0
            for gt in class_ground_truth:
                iou = calculate_iou(pred['bbox'], gt['bbox'])
                max_iou = max(max_iou, iou)
            ious.append(max_iou)
        
        avg_iou = np.mean(ious) if ious else 0
        print(f"{class_name} 平均IoU: {avg_iou:.3f}")
```

## 📝 最佳实践

### 1. 数据准备
- 确保两个类别都有足够的样本
- 检查标注质量
- 平衡训练集和验证集

### 2. 训练策略
- 使用适当的数据增强
- 监控类别平衡性
- 调整学习率和训练轮数

### 3. 评估方法
- 关注每个类别的性能
- 分析混淆矩阵
- 检查困难样本

### 4. 优化方向
- 根据性能分析调整策略
- 使用硬负样本挖掘
- 考虑类别权重调整

## 🎯 总结

多类别检测的关键点：

1. **类别配置**: 正确设置类别名称和索引
2. **性能监控**: 分别监控每个类别的性能
3. **平衡性**: 确保类别间的平衡
4. **困难样本**: 使用硬负样本挖掘提高性能
5. **调试分析**: 深入分析类别间的混淆情况

通过合理的配置和监控，可以实现高质量的多类别检测模型。











