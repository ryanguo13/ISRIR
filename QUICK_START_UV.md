# 🚀 快速开始指南 (uv版本)

## 使用uv的一键式训练设置

### 1. 安装uv并配置环境

```bash
# 一键安装uv并配置所有依赖
python install_dependencies_uv.py
```

### 2. 开始训练

```bash
# 方式1: 直接使用uv运行
uv run python run_optimized_training.py

# 方式2: 使用便捷脚本 (Windows)
run_training.bat

# 方式2: 使用便捷脚本 (Unix/Linux/macOS)
./run_training.sh

# 方式3: 使用Python脚本
python run_with_uv.py
```

### 3. 监控训练

- **本地监控**: `uv run tensorboard --logdir tb_logger`
- **云端监控**: 访问 [https://wandb.ai](https://wandb.ai) 查看实时训练状态

## 🎯 uv的优势

✅ **极速安装** - 比pip快10-100倍的包安装速度  
✅ **智能解析** - 更好的依赖冲突解决  
✅ **完全兼容** - 完美兼容pip和PyPI生态系统  
✅ **现代工具** - 基于Rust构建，更稳定可靠  

## 📁 项目结构

使用uv后，项目将包含：

```
├── pyproject.toml           # uv项目配置文件
├── uv.lock                  # 锁定的依赖版本
├── run_training.bat         # Windows启动脚本
├── run_training.sh          # Unix启动脚本
├── run_with_uv.py          # Python启动脚本
├── config/                  # 配置文件
├── core/                    # 核心代码
├── model/                   # 模型定义
├── param_outputs/           # 参数输出
└── ...
```

## ⚡ 常用uv命令

```bash
# 环境管理
uv shell                     # 激活虚拟环境
uv sync                      # 同步依赖到最新

# 包管理
uv add <package>             # 添加新包
uv add <package> --dev       # 添加开发依赖
uv remove <package>          # 移除包
uv list                      # 列出已安装的包

# 训练相关
uv run python run_optimized_training.py          # 基础训练
uv run python run_optimized_training.py --no-wandb  # 离线训练
uv run python run_optimized_training.py --dry-run   # 环境检查
```

## 🔧 高级配置

### 自定义Python版本

```bash
# 指定特定Python版本
uv init --python 3.11

# 或在pyproject.toml中修改
requires-python = ">=3.11"
```

### 添加GPU支持

```bash
# 手动添加CUDA版本的PyTorch
uv add torch torchvision torchaudio --index https://download.pytorch.org/whl/cu118
```

### 开发模式

```bash
# 安装开发依赖
uv sync --extra dev

# 格式化代码
uv run black .
uv run isort .
```

## 🚀 性能对比

| 操作 | pip | uv | 提升 |
|------|-----|----|----|
| 环境创建 | 30-60s | 3-5s | 6-12x |
| 依赖安装 | 60-180s | 5-15s | 10-30x |
| 依赖解析 | 慢 | 极快 | 显著提升 |
| 缓存利用 | 有限 | 智能 | 大幅提升 |

## 🔧 常见问题

**Q: uv和conda能一起使用吗？**  
A: 可以！uv可以在conda环境中使用，只需在conda环境激活后运行uv命令

**Q: 如何迁移现有的pip项目？**  
A: 运行 `uv init` 然后 `uv add` 现有的依赖即可

**Q: 网络问题怎么办？**  
A: uv支持代理设置和镜像源，可以在pyproject.toml中配置

**Q: 如何清理缓存？**  
A: 运行 `uv cache clean` 清理缓存

## 📚 更多资源

- [uv官方文档](https://docs.astral.sh/uv/)
- [uv GitHub仓库](https://github.com/astral-sh/uv)
- [Python项目管理最佳实践](https://packaging.python.org/)

开始享受超快速的Python环境管理和训练体验！ ⚡ 