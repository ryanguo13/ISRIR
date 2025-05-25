# 📦 手动安装指南

如果自动安装脚本失败，请按照以下步骤手动安装：

## 方法1: 直接下载uv (推荐)

### Windows

```powershell
# 方式1: 使用PowerShell (绕过执行策略)
powershell -ExecutionPolicy Bypass -c "irm https://astral.sh/uv/install.ps1 | iex"

# 方式2: 直接下载exe文件
# 访问 https://github.com/astral-sh/uv/releases/latest
# 下载 uv-x86_64-pc-windows-msvc.zip
# 解压并将uv.exe放到PATH中

# 方式3: 使用winget
winget install --id=astral-sh.uv -e
```

### macOS/Linux

```bash
# 使用官方安装脚本
curl -LsSf https://astral.sh/uv/install.sh | sh

# 或使用brew (macOS)
brew install uv

# 或使用cargo
cargo install uv
```

## 方法2: 使用现有的Python环境

如果uv安装有问题，可以直接使用当前的Python环境：

```bash
# 1. 安装基础依赖
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install numpy matplotlib wandb tensorboardx tqdm opencv-python pillow lmdb scipy scikit-image seaborn pandas

# 2. 直接运行训练
python run_optimized_training.py

# 3. 或者使用requirements文件
pip install -r requirements_optimized.txt
```

## 方法3: 使用conda

```bash
# 创建新环境
conda create -n sr-training python=3.9
conda activate sr-training

# 安装PyTorch
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# 安装其他依赖
pip install wandb tensorboardx opencv-python lmdb scipy scikit-image seaborn

# 运行训练
python run_optimized_training.py
```

## 验证安装

无论使用哪种方法，都可以运行以下命令验证：

```bash
# 如果使用uv
uv --version
uv run python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"

# 如果使用pip/conda
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"
python -c "import wandb, numpy, matplotlib; print('依赖检查通过')"
```

## 开始训练

配置完成后，使用以下命令之一开始训练：

```bash
# 使用uv (如果安装成功)
uv run python run_optimized_training.py

# 使用普通Python
python run_optimized_training.py

# 检查环境
python run_optimized_training.py --dry-run
```

## 常见问题解决

### 1. PowerShell执行策略问题
```powershell
# 临时允许脚本执行
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### 2. 网络问题
```bash
# 使用国内镜像 (如果在中国)
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple torch torchvision torchaudio
```

### 3. CUDA版本问题
```bash
# 检查CUDA版本
nvidia-smi

# 安装对应的PyTorch版本
# CUDA 11.8: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# CUDA 12.1: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
# CPU only: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### 4. 显存不足
如果显存不足，可以修改配置文件中的batch_size：

```json
{
  "datasets": {
    "train": {
      "batch_size": 2,  // 减小到2或1
      "num_workers": 4
    }
  }
}
```

配置完成后就可以开始高效的超分辨率训练了！ 🚀 