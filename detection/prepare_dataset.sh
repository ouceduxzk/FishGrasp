#!/usr/bin/env bash
set -euo pipefail

# 用途：统一的数据准备脚本
# - 若输入目录下发现COCO json（含images数组），将调用 detection/prepare_yolo_dataset.py 完成转换与切分
# - 否则，调用 detection/convert_json_to_yolo.py 将逐图JSON(两点矩形)转为YOLO TXT，然后在此脚本中完成随机切分
#
# 依赖：
#   detection/prepare_yolo_dataset.py
#   detection/convert_json_to_yolo.py
#
# 用法示例：
#   bash detection/prepare_dataset.sh \
#     --source ./raw_9.9_sum \
#     --out ./datasets/l0_9.9_sum \
#     --train 0.8 --val 0.1 --test 0.1 \
#     --copy
#   bash detection/prepare_dataset.sh --source ./raw_9.10 --out ./datasets/l0_9.10  --train 0.8 --val 0.1 --test 0.1 --copy
#
# 说明：
#   --source  根目录，要求包含 images/ 与 labels/ 子目录（逐图json场景），或包含COCO json
#   --out     输出数据集根目录
#   --train/--val/--test  切分比例（相加应为1.0）
#   --copy    拷贝图片（默认硬链接）
#   --symlink 软链接（与 --copy 互斥）

SOURCE=""
OUT=""
TRAIN=0.8
VAL=0.1
TEST=0.1
MODE="hardlink"   # hardlink | copy | symlink

while [[ $# -gt 0 ]]; do
  case "$1" in
    --source)
      SOURCE="$2"; shift 2;;
    --out)
      OUT="$2"; shift 2;;
    --train)
      TRAIN="$2"; shift 2;;
    --val)
      VAL="$2"; shift 2;;
    --test)
      TEST="$2"; shift 2;;
    --copy)
      MODE="copy"; shift 1;;
    --symlink)
      MODE="symlink"; shift 1;;
    -h|--help)
      echo "用法: $0 --source DIR --out DIR [--train 0.8 --val 0.1 --test 0.1] [--copy|--symlink]"; exit 0;;
    *) echo "未知参数: $1"; exit 1;;
  esac
done

if [[ -z "$SOURCE" || -z "$OUT" ]]; then
  echo "[错误] 需要 --source 与 --out"; exit 1
fi

SOURCE="$(realpath "$SOURCE")"
OUT="$(realpath "$OUT")"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

PY_PREPARE="$ROOT_DIR/detection/prepare_yolo_dataset.py"
PY_CONVERT="$ROOT_DIR/detection/convert_json_to_yolo.py"

mkdir -p "$OUT"

# 检查比例和是否为1.0（允许微小浮点误差）
sum_ratio=$(awk -v a="$TRAIN" -v b="$VAL" -v c="$TEST" 'BEGIN{printf "%.6f", a+b+c}')
awk -v s="$sum_ratio" 'BEGIN{if ((s<0.999 || s>1.001)) print "[警告] 切分比例之和不是1.0: " s}'

# 检查是否存在COCO json（含 images 字段）
find_coco_json() {
  local dir="$1"
  local found=""
  while IFS= read -r -d '' j; do
    if python3 - "$j" << 'PY'
import json,sys
p=sys.argv[1]
try:
    with open(p,'r',encoding='utf-8') as f:
        d=json.load(f)
    if isinstance(d,dict) and isinstance(d.get('images',None),list):
        sys.exit(0)
except Exception:
    pass
sys.exit(1)
PY
    then
      echo "$j"
      return 0
    fi
  done < <(find "$dir" -type f -name "*.json" -print0)
  return 1
}

COCO_JSON=""
if COCO_JSON=$(find_coco_json "$SOURCE"); then
  echo "[信息] 发现COCO标注: $COCO_JSON"
  # 直接使用 prepare_yolo_dataset.py 完成转换+切分
  CMD=(python3 "$PY_PREPARE" --source_dir "$SOURCE" --out_dir "$OUT" \
       --train_ratio "$TRAIN" --val_ratio "$VAL" --test_ratio "$TEST")
  if [[ "$MODE" == "copy" ]]; then CMD+=(--copy); fi
  if [[ "$MODE" == "symlink" ]]; then CMD+=(--symlink); fi
  echo "运行: ${CMD[*]}"
  "${CMD[@]}"
  echo "[完成] 已基于COCO完成YOLO转换与切分 -> $OUT"
  exit 0
fi

echo "[信息] 未发现COCO标注，使用逐图JSON转换流程(convert_json_to_yolo.py)"

# 1) 先将逐图json转为 YOLO TXT，并把所有图片集中到 staging 目录
STAGING="$OUT/_staging"
IMG_ALL="$STAGING/images_all"
LBL_ALL="$STAGING/labels_all"
mkdir -p "$IMG_ALL" "$LBL_ALL"

CONVERT_CMD=(python3 "$PY_CONVERT" --source "$SOURCE" --labels_out "$LBL_ALL" --images_out "$IMG_ALL")
if [[ "$MODE" == "copy" ]]; then CONVERT_CMD+=(--copy_images); fi
# split=mirror 仅镜像目录，不做集合拆分；此处保持默认 none，即平铺
CONVERT_CMD+=(--split none)

echo "运行: ${CONVERT_CMD[*]}"
"${CONVERT_CMD[@]}"

echo "[信息] 转换完成，开始随机切分 train/val/test"

# 2) 随机切分图片，并同步对应label
ensure_dirs() {
  for sub in images/train images/val images/test labels/train labels/val labels/test; do
    mkdir -p "$OUT/$sub"
  done
}
ensure_dirs

shopt -s nullglob
mapfile -t IMAGES < <(find "$IMG_ALL" -maxdepth 1 -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" -o -iname "*.bmp" -o -iname "*.tif" -o -iname "*.tiff" \) | sort -R)
TOTAL=${#IMAGES[@]}
if [[ "$TOTAL" -eq 0 ]]; then
  echo "[错误] 未在 $IMG_ALL 发现任何图片"; exit 1
fi

# 使用四舍五入计算样本数，并保证和为 TOTAL
n_train=$(python3 -c "import sys,math; T=int(sys.argv[1]); tr=float(sys.argv[2]); print(int(round(T*tr)))" "$TOTAL" "$TRAIN")
n_val=$(python3 -c   "import sys,math; T=int(sys.argv[1]); vr=float(sys.argv[2]); print(int(round(T*vr)))" "$TOTAL" "$VAL")

n_test=$(( TOTAL - n_train - n_val ))
if (( n_test < 0 )); then
  overflow=$(( -n_test ))
  if (( n_val >= overflow )); then
    n_val=$(( n_val - overflow ))
  else
    rem=$(( overflow - n_val ))
    n_val=0
    if (( n_train >= rem )); then
      n_train=$(( n_train - rem ))
    else
      n_train=0
    fi
  fi
  n_test=$(( TOTAL - n_train - n_val ))
fi

# 打印比例和数量
pct_train=$(python3 -c "import sys; T=int(sys.argv[1]); n=int(sys.argv[2]); print(f'{(n/max(1,T))*100:.2f}%')" "$TOTAL" "$n_train")
pct_val=$(python3   -c "import sys; T=int(sys.argv[1]); n=int(sys.argv[2]); print(f'{(n/max(1,T))*100:.2f}%')" "$TOTAL" "$n_val")
pct_test=$(python3  -c "import sys; T=int(sys.argv[1]); n=int(sys.argv[2]); print(f'{(n/max(1,T))*100:.2f}%')" "$TOTAL" "$n_test")

echo "[信息] 总图片数: $TOTAL"
echo "[信息] 计划比例: train=$TRAIN, val=$VAL, test=$TEST"
echo "[信息] 实际分配: train=$n_train ($pct_train), val=$n_val ($pct_val), test=$n_test ($pct_test)"

split_and_place() {
  local img="$1" split="$2"
  local base="$(basename "$img")"
  local stem="${base%.*}"
  local src_lbl="$LBL_ALL/$stem.txt"
  local dst_img="$OUT/images/$split/$base"
  local dst_lbl="$OUT/labels/$split/$stem.txt"

  if [[ "$MODE" == "copy" ]]; then
    cp -f "$img" "$dst_img" || return 0
  elif [[ "$MODE" == "symlink" ]]; then
    ln -sf "$img" "$dst_img" || cp -f "$img" "$dst_img"
  else
    ln -f "$img" "$dst_img" 2>/dev/null || cp -f "$img" "$dst_img"
  fi
  if [[ -f "$src_lbl" ]]; then
    cp -f "$src_lbl" "$dst_lbl"
  else
    # 若无标签，写空文件
    : > "$dst_lbl"
  fi
}

idx=0
for img in "${IMAGES[@]}"; do
  if   [[ $idx -lt $n_train ]]; then split_and_place "$img" train
  elif [[ $idx -lt $((n_train+n_val)) ]]; then split_and_place "$img" val
  else split_and_place "$img" test
  fi
  idx=$((idx+1))
  if (( idx % 100 == 0 )); then echo "  进度: $idx/$TOTAL"; fi
done

echo "[信息] 切分完成: train=$n_train, val=$n_val, test=$n_test"

# 3) 生成 dataset.yaml，尽量复用convert产生的类别顺序（若有）
NAMES_LINE=""
FULL_YAML="$STAGING/dataset_full.yaml"
if python3 "$PY_CONVERT" --source "$SOURCE" --labels_out "$LBL_ALL" --images_out "$IMG_ALL" --class_order "" --names_out "$FULL_YAML" >/dev/null 2>&1; then
  if [[ -f "$FULL_YAML" ]]; then
    NAMES_LINE=$(grep -E '^names:' "$FULL_YAML" || true)
  fi
fi
if [[ -z "$NAMES_LINE" ]]; then
  # 回退：空类别占位
  NAMES_LINE="names: []"
fi

cat > "$OUT/dataset.yaml" <<YAML
path: $OUT
train: images/train
val: images/val
test: images/test
$NAMES_LINE
YAML

echo "[OK] 生成数据集配置: $OUT/dataset.yaml"

# 4) 清理中间目录：硬链/拷贝模式可安全删除；软链模式需保留
if [[ "$MODE" == "symlink" ]]; then
  echo "[提示] 当前为symlink模式，为保持链接有效，将保留中间目录: $STAGING"
else
  rm -rf "$STAGING" || true
  echo "[清理] 已删除中间目录: $STAGING"
fi

echo "[完成] 数据集已准备就绪 -> $OUT"
