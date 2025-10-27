# 项目简介

阅读《深度学习入门：基于Python的理论与实现》时，实现的代码。

官方示例代码：[deep-learning-from-scratch](https://github.com/oreilly-japan/deep-learning-from-scratch)

# 环境配置

## 操作系统

win11, x64

## 安装Miniconda 3

安装Miniconda 3，其中包括Python和Conda，版本如下：
- Python version：3.13.5.final.0
- Conda version：25.5.1

## 添加Miniconda路径到环境变量中

在系统环境变量中，添加Miniconda路径：

```
XXX\miniconda3
XXX\miniconda3\Scripts
```

## 创建并激活虚拟环境

- 创建虚拟环境：
    ```
    conda create -n DeepLearningFromScratch
    ```

- 查询虚拟环境：
    ```
    conda info --envs
    ```

- 激活虚拟环境：
    ```
    conda activate DeepLearningFromScratch
    ```

## 安装Python依赖

在`DeepLearningFromScratch`虚拟环境中，安装依赖：
```
conda env update -f environment.yml
```

## 配置VS Code

- 在操作系统中，设置环境变量`CONDA_ROOT`：指向Miniconda的安装路径
    - 在项目的`.vscode\settings.json`中会用到这个环境变量

- 新建Terminal，默认会打开`Anaconda Prompt (DeepLearningFromScratch)`，相当于打开`Anaconda Prompt`，并激活`DeepLearningFromScratch`虚拟环境
    - 对应的配置在`.vscode\settings.json`中


- 选择Python解释器：`Ctrl+Shift+P` -> `Python: Select Interpreter` -> `Python 3.13.5 (DeepLearningFromScratch)`
