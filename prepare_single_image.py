import os
from PIL import Image
from pathlib import Path
import data.util as Util

def resize_and_convert(img, size, resample=Image.BICUBIC):
    """将图片调整到指定尺寸"""
    if img.size[0] != size or img.size[1] != size:
        # 先resize到指定尺寸，然后center crop
        img = img.resize((size, size), resample)
    return img

def prepare_single_image_data(input_dir, output_dir, lr_size=16, hr_size=128):
    """
    准备单张图片的数据
    input_dir: 输入图片目录 (data/singlepic)
    output_dir: 输出目录 (data/single_image_infer)
    lr_size: 低分辨率尺寸 (16)
    hr_size: 高分辨率尺寸 (128)
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f'{output_dir}/lr_{lr_size}', exist_ok=True)
    os.makedirs(f'{output_dir}/hr_{hr_size}', exist_ok=True)
    os.makedirs(f'{output_dir}/sr_{lr_size}_{hr_size}', exist_ok=True)
    
    # 获取所有图片文件
    input_path = Path(input_dir)
    image_files = [f for f in input_path.glob('*') if Util.is_image_file(f.name)]
    
    print(f"找到 {len(image_files)} 张图片需要处理")
    
    for idx, img_file in enumerate(image_files):
        print(f"正在处理: {img_file.name}")
        
        # 读取原始图片
        img = Image.open(img_file).convert('RGB')
        
        # 生成低分辨率图片 (16x16)
        lr_img = resize_and_convert(img, lr_size)
        
        # 生成高分辨率图片 (128x128) 
        hr_img = resize_and_convert(img, hr_size)
        
        # 生成超分辨率参考图片 (从16x16双三次插值到128x128)
        sr_img = resize_and_convert(lr_img, hr_size)
        
        # 保存图片，使用文件名（去掉扩展名）
        base_name = img_file.stem
        
        lr_img.save(f'{output_dir}/lr_{lr_size}/{base_name}.png')
        hr_img.save(f'{output_dir}/hr_{hr_size}/{base_name}.png')
        sr_img.save(f'{output_dir}/sr_{lr_size}_{hr_size}/{base_name}.png')
        
        print(f"  保存了 lr_{lr_size}/{base_name}.png")
        print(f"  保存了 hr_{hr_size}/{base_name}.png") 
        print(f"  保存了 sr_{lr_size}_{hr_size}/{base_name}.png")

if __name__ == '__main__':
    input_dir = 'data/singlepic'
    output_dir = 'data/single_image_infer'
    
    print("开始准备单张图片数据...")
    prepare_single_image_data(input_dir, output_dir)
    print("数据准备完成！") 