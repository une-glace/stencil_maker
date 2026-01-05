import cv2
import numpy as np
import os
import math

# 配置参数
INPUT_FILE = 'ori.png'
OUTPUT_DIR = 'stencil_output'
TARGET_SIZE_CM = (150, 150)  # 1.5m x 1.5m
DPI = 72  # 打印分辨率，72dpi对于大尺寸涂鸦模板足够，甚至可以更低，方便处理
# A4 纸张像素尺寸 (at 72 DPI)
A4_WIDTH_PX = int(21.0 / 2.54 * DPI)
A4_HEIGHT_PX = int(29.7 / 2.54 * DPI)
NUM_CLUSTERS = 4  # 颜色数量：背景黄，浅灰，深灰，黑

def create_test_image():
    """创建一个测试图片，如果原图不存在"""
    print("未找到 ori.png，生成测试图片...")
    img = np.zeros((1000, 1000, 3), dtype=np.uint8)
    img[:] = (75, 143, 157) # 背景黄 (BGR)
    cv2.circle(img, (500, 500), 400, (100, 100, 100), -1) # 深灰
    cv2.circle(img, (500, 500), 300, (200, 200, 200), -1) # 浅灰
    cv2.putText(img, "TEST", (350, 550), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 0), 10) # 黑字
    cv2.imwrite(INPUT_FILE, img)
    print(f"测试图片已保存为 {INPUT_FILE}")

def ensure_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)

def quantization(image, k):
    """使用 K-Means 减少颜色数量"""
    data = np.float32(image).reshape((-1, 3))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, centers = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    result = centers[labels.flatten()]
    return result.reshape(image.shape), centers, labels.reshape(image.shape[:2])

def sort_colors_by_brightness(centers):
    """按亮度排序颜色，返回排序后的索引"""
    # 计算亮度: 0.299*R + 0.587*G + 0.114*B (注意 OpenCV 是 BGR)
    brightness = np.sum(centers * np.array([0.114, 0.587, 0.299]), axis=1)
    return np.argsort(brightness)[::-1] # 从亮到暗

def slice_and_save(layer_img, layer_name, scale_factor=1.0):
    """将大图切分为 A4 大小的块"""
    h, w = layer_img.shape
    
    # 打印时，我们需要让图像的物理尺寸对应 1.5m
    # 现在的像素 w, h 对应 150cm
    # 计算需要多少张 A4 纸
    # A4 宽约 21cm, 高 29.7cm (减去页边距，有效区域按 19x27算比较安全)
    
    # 为了简化，我们假设 user 打印时选择 "100% 缩放" 或 "海报打印"
    # 这里我们直接把图片切成小块保存
    
    # 计算行数和列数
    # 150cm / 19cm = ~7.8 -> 8张
    effective_w_cm = 19.0
    effective_h_cm = 27.7
    
    pixels_per_cm = w / 150.0
    
    chunk_w = int(effective_w_cm * pixels_per_cm)
    chunk_h = int(effective_h_cm * pixels_per_cm)
    
    rows = math.ceil(h / chunk_h)
    cols = math.ceil(w / chunk_w)
    
    layer_dir = os.path.join(OUTPUT_DIR, layer_name)
    ensure_directory(layer_dir)
    
    print(f"正在切片 {layer_name}: {rows}行 x {cols}列 ...")
    
    for r in range(rows):
        for c in range(cols):
            x1 = c * chunk_w
            y1 = r * chunk_h
            x2 = min(x1 + chunk_w, w)
            y2 = min(y1 + chunk_h, h)
            
            chunk = layer_img[y1:y2, x1:x2]
            
            # 增加白色边框方便粘贴，或者直接保存内容
            # 这里为了省墨，背景是白的，内容是黑的
            # 注意：传入的 layer_img 已经是黑底白字还是白底黑字？
            # 通常模板：黑色是需要挖空的地方。打印时：黑色费墨。
            # 建议：打印出轮廓线（节省墨水）或者 灰色填充。
            # 这里直接保存二值化图（黑=墨水=挖空）。
            
            # 如果这块是全白的（不需要挖），可以跳过，但为了拼接完整最好保留
            if np.mean(chunk) == 255: 
                # 全白，甚至可以不打印，但为了编号方便还是存一下
                pass

            # 添加文件名编号
            filename = f"{layer_name}_R{r+1}_C{c+1}.png"
            
            # 加上边框和编号文字方便拼接
            chunk_with_border = cv2.copyMakeBorder(chunk, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=128)
            # 转彩色以便写红字
            chunk_color = cv2.cvtColor(chunk_with_border, cv2.COLOR_GRAY2BGR)
            cv2.putText(chunk_color, f"{layer_name} [{r+1},{c+1}]", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            cv2.imwrite(os.path.join(layer_dir, filename), chunk_color)

def main():
    if not os.path.exists(INPUT_FILE):
        create_test_image()
        
    print(f"读取图片: {INPUT_FILE}")
    img = cv2.imread(INPUT_FILE)
    if img is None:
        print("错误：无法读取图片")
        return

    # 1. 调整大小
    # 目标尺寸 150cm，按 20 px/cm (约50dpi) 计算 -> 3000px
    # 这样既保证细节又不会太大
    target_px = 3000
    h, w = img.shape[:2]
    scale = target_px / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    img_resized = cv2.resize(img, (new_w, new_h))
    print(f"调整尺寸为: {new_w}x{new_h}")

    # 2. 模糊处理去噪 (平滑细节，方便做模板)
    print("正在进行双边滤波去噪...")
    img_blur = cv2.bilateralFilter(img_resized, 9, 75, 75)

    # 3. 颜色量化
    print(f"正在聚类分析 {NUM_CLUSTERS} 种颜色...")
    quantized_img, centers, labels = quantization(img_blur, NUM_CLUSTERS)
    
    # 保存预览图
    ensure_directory(OUTPUT_DIR)
    cv2.imwrite(os.path.join(OUTPUT_DIR, 'preview_quantized.jpg'), quantized_img)
    print("已保存色彩预览图: preview_quantized.jpg")

    # 4. 排序颜色 (亮 -> 暗)
    sorted_indices = sort_colors_by_brightness(centers)
    
    # 5. 分层提取
    # indices: 0=最亮(背景), 1=浅灰, 2=深灰, 3=最暗(黑)
    # 既然是墙绘，通常背景色直接刷墙。
    # 所以我们需要制作的是：
    # Layer 1: 浅灰层 (把浅灰、深灰、黑的地方都挖空？不，通常是一层压一层)
    # 方案 A (覆盖法): 
    #   Step 0: 墙刷黄 (Color 0)
    #   Step 1: 刷浅灰 (Color 1) -> 模板是 (Color 1 + 2 + 3) 的区域
    #   Step 2: 刷深灰 (Color 2) -> 模板是 (Color 2 + 3) 的区域
    #   Step 3: 刷黑色 (Color 3) -> 模板是 (Color 3) 的区域
    # 这样可以保证边缘严丝合缝，不会漏底。
    
    layer_names = ['0_Background_Yellow', '1_LightGrey', '2_DarkGrey', '3_Black']
    
    print("开始分层处理...")
    
    # 累积Mask：用于覆盖法
    # 比如做浅灰层时，所有比浅灰暗的区域也要包含进来，因为之后会被更暗的颜色盖住
    # 这样模板更连贯，不容易碎
    
    # 我们的 labels 是二维数组，值是 0,1,2,3 (对应 sorted_indices 里的颜色)
    # 但 labels 里的值是 kmeans 随机生成的索引，不是按亮度排的
    # 我们需要一个映射： label_val -> brightness_rank (0=lightest)
    
    rank_map = { original_idx: rank for rank, original_idx in enumerate(sorted_indices) }
    
    # 创建一个 rank 矩阵
    h, w = labels.shape
    rank_mat = np.zeros((h, w), dtype=np.uint8)
    for r in range(h):
        for c in range(w):
            rank_mat[r, c] = rank_map[labels[r, c]]
            
    # 生成图层
    # Layer 0 是背景，不需要模板
    
    for i in range(1, NUM_CLUSTERS):
        # 目标：生成第 i 层模板
        # 这一层需要涂颜色的区域是：亮度等级 >= i 的所有区域
        # (因为深色盖浅色，画深灰的时候，黑色区域也先涂成深灰也没关系，反正最后会盖黑)
        # 这样模板面积大，更好刻，不容易断。
        
        mask = np.where(rank_mat >= i, 0, 255).astype(np.uint8) # 0是黑(墨水/挖空)，255是白(纸)
        
        # 边缘检测，提取轮廓 (可选，如果用户只想要轮廓线)
        # 这里我们直接给实心块，用户打印出来可以直接贴在厚纸上，把黑的地方刻掉
        
        # 也可以做一个轮廓版，省墨
        edges = cv2.Canny(mask, 100, 200)
        # 反转 edges: 边缘是黑，背景白
        edges_inv = 255 - edges
        
        # 保存整张预览
        layer_name = layer_names[i]
        save_path = os.path.join(OUTPUT_DIR, f'preview_{layer_name}.png')
        cv2.imwrite(save_path, mask)
        
        # 切片
        # 这里我们切片保存 mask (实心黑块)，方便用户看哪里要挖
        slice_and_save(mask, layer_name)
        
    print("处理完成！请查看 stencil_output 文件夹。")

if __name__ == '__main__':
    main()
