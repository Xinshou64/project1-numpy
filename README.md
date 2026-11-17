# Project 1：NumPy 图像编码器
这个项目使用 NumPy 处理图片，为每张图片生成 512 个特征数值。
## 准备文件
先解压提供的“project1-学生版”压缩包，再将本仓库中的代码文件放入解压后的项目文件夹，替换同名文件。保留原来的 `weights` 和 `assets` 文件夹。
## 安装需要的库
使用 Python 3.9 或以上版本，在项目文件夹中打开终端，运行：
```powershell
python -m pip install -r requirements.txt
```
## 检查代码
```powershell
python check.py --stage all
```
各项显示 `PASS`，表示检查通过。
## 运行程序
```powershell
python run.py --output output_tokens.npz
```
程序会处理老师提供的十张图片，并生成 `output_tokens.npz` 文件。运行完成后会显示：
```text
Saved (10, 512) to output_tokens.npz
```
文件中保存了图片编号和每张图片的 512 个特征数值。
## 完成内容
补全了 `image_encoder.py` 中的六组任务，包括图片切块、特征转换、注意力计算和最终结果输出。其余辅助代码和数据已提供。完成过程中参考了 AI 的解释和代码建议。

