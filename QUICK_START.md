# 🚀 快速开始指南

## 一键式训练设置

### 1. 安装依赖

```bash
# 自动安装所有必要依赖
python install_dependencies.py
```

### 2. 开始训练

```bash
# 自动优化配置并开始训练
python run_optimized_training.py
```

### 3. 监控训练

- **本地监控**: 打开浏览器访问 tensorboard
- **云端监控**: 访问 [https://wandb.ai](https://wandb.ai) 查看实时训练状态

## 🎯 主要优化功能

✅ **2倍训练速度** - 混合精度训练 + AdamW优化器  
✅ **自动配置** - 根据GPU内存自动调整批量大小  
✅ **完整监控** - Wandb实时跟踪 + 参数输出  
✅ **断点恢复** - 支持从任意检查点继续训练  

## 📁 输出结果

训练完成后，你将获得：

- `checkpoint/` - 模型检查点文件
- `results/` - 生成的超分辨率图像
- `param_outputs/` - 详细的训练参数分析
- `logs/` - 训练日志文件

## ⚡ 常用命令

```bash
# 从检查点恢复训练
python run_optimized_training.py --resume checkpoint/I640000_E37

# 禁用wandb (离线训练)
python run_optimized_training.py --no-wandb

# 仅检查环境
python run_optimized_training.py --dry-run

# 自定义配置
python run_optimized_training.py --config your_config.json
```

## 🔧 常见问题

**Q: 显存不足怎么办？**  
A: 脚本会自动根据显存调整批量大小，如仍不足可手动编辑配置文件减小 `batch_size`

**Q: 没有GPU可以训练吗？**  
A: 可以，但速度会很慢。建议使用Google Colab或其他云GPU服务

**Q: 如何查看训练进度？**  
A: 登录 wandb.ai 可以实时查看损失曲线、生成图像等

开始享受你的高效训练体验吧！ 🎉 