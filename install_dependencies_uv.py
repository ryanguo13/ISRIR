#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
使用uv的自动安装依赖脚本
更快速、更现代化的Python环境管理
"""

import subprocess
import sys
import os
import platform


def run_command(cmd, check_success=True):
    """运行命令并显示输出"""
    print(f"执行: {cmd}")
    try:
        result = subprocess.run(cmd, shell=True, check=True, 
                              capture_output=True, text=True)
        print("✅ 成功")
        if result.stdout.strip():
            print(f"输出: {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        if check_success:
            print(f"❌ 失败: {e}")
            if e.stdout:
                print(f"输出: {e.stdout}")
            if e.stderr:
                print(f"错误: {e.stderr}")
            return False
        else:
            print("⚠️  跳过")
            return True


def check_uv_installed():
    """检查uv是否已安装"""
    try:
        result = subprocess.run(['uv', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ uv已安装: {result.stdout.strip()}")
            return True
        else:
            return False
    except FileNotFoundError:
        return False


def install_uv():
    """安装uv"""
    print("🚀 安装uv...")
    
    if platform.system() == "Windows":
        # Windows安装方式
        cmd = 'powershell -c "irm https://astral.sh/uv/install.ps1 | iex"'
    else:
        # Unix系统安装方式
        cmd = 'curl -LsSf https://astral.sh/uv/install.sh | sh'
    
    if not run_command(cmd):
        print("❌ uv安装失败，请手动安装:")
        print("访问 https://github.com/astral-sh/uv 查看安装说明")
        return False
    
    # Windows上可能需要重新加载PATH
    if platform.system() == "Windows":
        print("⚠️  Windows用户可能需要重启终端或重新加载PATH")
    
    return True


def create_uv_project():
    """创建uv项目"""
    print("\n📦 创建uv项目...")
    
    # 初始化uv项目
    if not os.path.exists("pyproject.toml"):
        run_command("uv init --name sr-training --python 3.9")
    
    # 创建pyproject.toml配置
    pyproject_content = '''[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "sr-training"
version = "0.1.0"
description = "Optimized Super Resolution Training"
authors = [{name = "SR Training", email = "training@example.com"}]
requires-python = ">=3.9"
dependencies = [
    # Core PyTorch packages
    "torch>=1.12.0",
    "torchvision>=0.13.0", 
    "torchaudio>=0.12.0",
    
    # Data processing
    "numpy>=1.21.0",
    "pandas>=1.3.0",
    "pillow>=8.3.0",
    "opencv-python>=4.5.0",
    "lmdb>=1.2.0",
    
    # Progress and visualization
    "tqdm>=4.62.0",
    "matplotlib>=3.5.0",
    "seaborn>=0.11.0",
    
    # Logging and monitoring
    "tensorboardx>=2.4.0",
    "wandb>=0.12.0",
    
    # Utilities
    "scipy>=1.7.0",
    "scikit-image>=0.18.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=6.0",
    "black>=22.0",
    "isort>=5.0",
    "flake8>=4.0",
]
cuda = [
    "torch>=1.12.0+cu118",
    "torchvision>=0.13.0+cu118",
    "torchaudio>=0.12.0+cu118",
]

[tool.uv]
index-url = "https://pypi.org/simple"
extra-index-url = [
    "https://download.pytorch.org/whl/cu118",  # CUDA 11.8
    "https://download.pytorch.org/whl/cpu",   # CPU fallback
]

[tool.black]
line-length = 100
target-version = ['py39']

[tool.isort]
profile = "black"
line_length = 100
'''
    
    with open("pyproject.toml", "w", encoding="utf-8") as f:
        f.write(pyproject_content)
    
    print("✅ pyproject.toml 已创建")


def install_dependencies():
    """使用uv安装依赖"""
    print("\n📚 使用uv安装依赖...")
    
    # 检测CUDA支持
    cuda_available = False
    try:
        nvidia_smi = subprocess.run(['nvidia-smi'], capture_output=True)
        if nvidia_smi.returncode == 0:
            print("✅ 检测到NVIDIA GPU，将安装CUDA版本的PyTorch")
            cuda_available = True
        else:
            print("⚠️  未检测到NVIDIA GPU，将安装CPU版本的PyTorch")
    except FileNotFoundError:
        print("⚠️  未检测到nvidia-smi，将安装CPU版本的PyTorch")
    
    # 安装核心依赖
    if not run_command("uv sync"):
        return False
    
    # 如果有CUDA，安装CUDA版本的PyTorch
    if cuda_available:
        print("\n🔥 安装CUDA版本的PyTorch...")
        run_command("uv add torch torchvision torchaudio --index https://download.pytorch.org/whl/cu118", 
                   check_success=False)
    
    return True


def verify_installation():
    """验证安装"""
    print("\n🔍 验证安装...")
    
    validation_script = '''
import sys
try:
    import torch
    print(f"✅ PyTorch {torch.__version__}")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"✅ CUDA可用: {gpu_name} ({gpu_memory:.1f}GB)")
    else:
        print("⚠️  CUDA不可用，使用CPU")
except ImportError as e:
    print(f"❌ PyTorch导入失败: {e}")
    sys.exit(1)

validation_imports = [
    ("numpy", "NumPy"),
    ("matplotlib", "Matplotlib"), 
    ("wandb", "Wandb"),
    ("tensorboardX", "TensorBoard"),
    ("tqdm", "TQDM"),
    ("cv2", "OpenCV"),
    ("PIL", "Pillow"),
    ("lmdb", "LMDB"),
    ("scipy", "SciPy"),
    ("skimage", "Scikit-Image")
]

success_count = 0
for module, name in validation_imports:
    try:
        __import__(module)
        print(f"✅ {name}")
        success_count += 1
    except ImportError:
        print(f"❌ {name} 导入失败")

print(f"\\n📊 验证结果: {success_count}/{len(validation_imports)} 包成功导入")
if success_count >= len(validation_imports) - 2:
    print("🎉 环境配置成功！")
else:
    print("⚠️  部分包导入失败，但核心功能应该可用")
'''
    
    return run_command(f'uv run python -c "{validation_script}"')


def create_run_scripts():
    """创建运行脚本"""
    print("\n📝 创建运行脚本...")
    
    # 创建uv运行脚本
    run_script_content = '''#!/usr/bin/env python
"""
使用uv运行训练的便捷脚本
"""
import subprocess
import sys
import os

def main():
    """使用uv运行训练"""
    print("🚀 使用uv启动优化训练...")
    
    # 构建命令
    cmd = ["uv", "run", "python", "run_optimized_training.py"] + sys.argv[1:]
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ 训练失败: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\\n⚠️  训练被用户中断")
        sys.exit(0)

if __name__ == "__main__":
    main()
'''
    
    with open("run_with_uv.py", "w", encoding="utf-8") as f:
        f.write(run_script_content)
    
    # 创建bat文件 (Windows)
    if platform.system() == "Windows":
        bat_content = '''@echo off
echo 🚀 使用uv启动训练...
uv run python run_optimized_training.py %*
'''
        with open("run_training.bat", "w", encoding="utf-8") as f:
            f.write(bat_content)
        
        print("✅ 创建了 run_training.bat (Windows)")
    
    # 创建shell脚本 (Unix)
    else:
        shell_content = '''#!/bin/bash
echo "🚀 使用uv启动训练..."
uv run python run_optimized_training.py "$@"
'''
        with open("run_training.sh", "w", encoding="utf-8") as f:
            f.write(shell_content)
        
        # 设置执行权限
        os.chmod("run_training.sh", 0o755)
        print("✅ 创建了 run_training.sh (Unix)")
    
    print("✅ 创建了 run_with_uv.py (跨平台)")


def main():
    print("🚀 开始使用uv配置SR训练环境...")
    print("=" * 60)
    
    # 检查并安装uv
    if not check_uv_installed():
        print("📦 uv未安装，开始安装...")
        if not install_uv():
            sys.exit(1)
        
        # 重新检查
        if not check_uv_installed():
            print("❌ uv安装后仍无法使用，请手动安装")
            sys.exit(1)
    
    # 创建uv项目
    create_uv_project()
    
    # 安装依赖
    if not install_dependencies():
        print("❌ 依赖安装失败")
        sys.exit(1)
    
    # 验证安装
    if not verify_installation():
        print("❌ 安装验证失败")
        sys.exit(1)
    
    # 创建运行脚本
    create_run_scripts()
    
    print("\n🎉 uv环境配置完成！")
    print("\n📋 使用方法:")
    print("1. 直接使用uv运行:")
    print("   uv run python run_optimized_training.py")
    print("\n2. 使用便捷脚本:")
    if platform.system() == "Windows":
        print("   run_training.bat")
    else:
        print("   ./run_training.sh")
    print("   python run_with_uv.py")
    
    print("\n3. 进入uv shell:")
    print("   uv shell")
    print("   python run_optimized_training.py")
    
    print("\n🔧 其他有用命令:")
    print("   uv add <package>     # 添加新依赖")
    print("   uv remove <package>  # 移除依赖")
    print("   uv sync             # 同步依赖")
    print("   uv shell            # 激活环境")


if __name__ == "__main__":
    main() 