import cv2
import numpy as np
import os
import math

# 配置参数
INPUT_FILE = 'ori.png'
OUTPUT_DIR = 'stencil_output'
TARGET_SIZE_CM = (150, 150)
CHUNK_SIZE_CM = (50, 50)  # 单个切片/纸张大小
DPI = 72  # 打印分辨率

NUM_CLUSTERS = 6  # 颜色数量：增加到6层以获得更细腻的过渡

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

def save_layer_image(layer_img, layer_name):
    """保存图层图像，如果尺寸超过 CHUNK_SIZE_CM 则自动切片"""
    layer_dir = os.path.join(OUTPUT_DIR, layer_name)
    ensure_directory(layer_dir)
    
    h, w = layer_img.shape[:2]
    
    # 计算像素密度
    # 假设图像已经按照 TARGET_SIZE_CM 缩放好了
    # 使用 float 避免整除误差
    pixels_per_cm_w = w / TARGET_SIZE_CM[0]
    pixels_per_cm_h = h / TARGET_SIZE_CM[1]
    
    # 计算切片像素大小
    chunk_px_w = int(CHUNK_SIZE_CM[0] * pixels_per_cm_w)
    chunk_px_h = int(CHUNK_SIZE_CM[1] * pixels_per_cm_h)
    
    # 计算需要切多少行多少列
    # 使用 ceil 向上取整，确保覆盖全图
    n_cols = math.ceil(w / chunk_px_w)
    n_rows = math.ceil(h / chunk_px_h)
    
    print(f"图层 {layer_name}: 总尺寸 {w}x{h} px, 切片大小 {chunk_px_w}x{chunk_px_h} px, 切分为 {n_rows}x{n_cols}={n_rows*n_cols} 张")

    for r in range(n_rows):
        for c in range(n_cols):
            # 计算当前切片的坐标范围
            x_start = c * chunk_px_w
            y_start = r * chunk_px_h
            x_end = min(x_start + chunk_px_w, w)
            y_end = min(y_start + chunk_px_h, h)
            
            # 裁剪图像
            chunk = layer_img[y_start:y_end, x_start:x_end]
            
            # 如果切出来的图是空的（理论上不会），跳过
            if chunk.size == 0:
                continue

            # 构造文件名: LayerName_Row_Col.png
            # 为了方便排序，使用 01, 02 格式
            filename = f"{layer_name}_R{r+1}_C{c+1}.png"
            save_path = os.path.join(layer_dir, filename)
            
            # 增加白色边框方便打印/粘贴
            chunk_with_border = cv2.copyMakeBorder(chunk, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=255)
            
            # 添加辅助信息文字
            img_color = cv2.cvtColor(chunk_with_border, cv2.COLOR_GRAY2BGR)
            
            info_text = f"{layer_name} | Row {r+1}/{n_rows} Col {c+1}/{n_cols}"
            cv2.putText(img_color, info_text, (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.putText(img_color, f"Size: 50cm x 50cm", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
            
            cv2.imwrite(save_path, img_color)
            # print(f"  已保存: {filename}")

def main():
    if not os.path.exists(INPUT_FILE):
        create_test_image()
        
    print(f"读取图片: {INPUT_FILE}")
    img = cv2.imread(INPUT_FILE)
    if img is None:
        print("错误：无法读取图片")
        return

    # 1. 调整大小
    # 目标尺寸按 20 px/cm (约50dpi) 计算
    # 这样既保证细节又不会太大
    target_px = int(max(TARGET_SIZE_CM) * 20)
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
    
    layer_names = [
        '0_Background_Yellow', 
        '1_HighLightGrey', 
        '2_LightGrey', 
        '3_MediumGrey', 
        '4_DarkGrey', 
        '5_Black'
    ]
    
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
        save_layer_image(mask, layer_name)
        
    print(f"处理完成！请查看 {OUTPUT_DIR} 文件夹。")

if __name__ == '__main__':
    main()
