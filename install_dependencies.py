#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
自动安装依赖脚本
"""

import subprocess
import sys
import os


def run_command(cmd):
    """运行命令并显示输出"""
    print(f"执行: {cmd}")
    try:
        result = subprocess.run(cmd, shell=True, check=True, 
                              capture_output=True, text=True)
        print("✅ 成功")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ 失败: {e}")
        print(f"输出: {e.stdout}")
        print(f"错误: {e.stderr}")
        return False


def main():
    print("🚀 开始安装优化训练依赖...")
    print("=" * 60)
    
    # 升级pip
    print("\n📦 升级pip...")
    run_command(f"{sys.executable} -m pip install --upgrade pip")
    
    # 安装PyTorch (根据系统自动选择)
    print("\n🔥 安装PyTorch...")
    
    # 检测CUDA
    try:
        import subprocess
        nvidia_smi = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if nvidia_smi.returncode == 0:
            print("✅ 检测到NVIDIA GPU，安装CUDA版本的PyTorch")
            # 安装CUDA版本
            torch_cmd = f"{sys.executable} -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118"
        else:
            print("⚠️  未检测到NVIDIA GPU，安装CPU版本的PyTorch")
            torch_cmd = f"{sys.executable} -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu"
    except:
        print("⚠️  无法检测GPU状态，安装默认版本的PyTorch")
        torch_cmd = f"{sys.executable} -m pip install torch torchvision torchaudio"
    
    if not run_command(torch_cmd):
        print("❌ PyTorch安装失败，请手动安装")
        return False
    
    # 安装其他依赖
    print("\n📚 安装其他依赖...")
    other_packages = [
        "numpy>=1.21.0",
        "pandas>=1.3.0", 
        "pillow>=8.3.0",
        "opencv-python>=4.5.0",
        "lmdb>=1.2.0",
        "tqdm>=4.62.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "tensorboardx>=2.4.0",
        "wandb>=0.12.0",
        "scipy>=1.7.0",
        "scikit-image>=0.18.0"
    ]
    
    for package in other_packages:
        print(f"\n安装 {package}...")
        if not run_command(f"{sys.executable} -m pip install {package}"):
            print(f"⚠️  {package} 安装失败，继续安装其他包...")
    
    # 可选包
    print("\n🚀 安装可选优化包...")
    optional_packages = ["ninja"]
    
    for package in optional_packages:
        print(f"\n安装 {package}...")
        run_command(f"{sys.executable} -m pip install {package}")
    
    # 验证安装
    print("\n🔍 验证安装...")
    validation_imports = [
        ("torch", "PyTorch"),
        ("torchvision", "TorchVision"),
        ("numpy", "NumPy"),
        ("matplotlib", "Matplotlib"),
        ("wandb", "Wandb"),
        ("tensorboardX", "TensorBoard"),
        ("tqdm", "TQDM"),
        ("cv2", "OpenCV"),
        ("PIL", "Pillow"),
        ("lmdb", "LMDB")
    ]
    
    success_count = 0
    for module, name in validation_imports:
        try:
            __import__(module)
            print(f"✅ {name}")
            success_count += 1
        except ImportError:
            print(f"❌ {name} 导入失败")
    
    print(f"\n📊 安装结果: {success_count}/{len(validation_imports)} 包成功安装")
    
    if success_count >= len(validation_imports) - 2:  # 允许1-2个可选包失败
        print("\n🎉 依赖安装完成！可以开始训练了。")
        print("\n▶️  运行以下命令开始训练:")
        print("python run_optimized_training.py")
        return True
    else:
        print("\n❌ 安装未完全成功，请检查错误信息")
        return False


if __name__ == "__main__":
    main() 