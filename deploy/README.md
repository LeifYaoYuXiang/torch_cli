# 0. Description 项目介绍
PyTorch的主要接口是Python编程语言。尽管Python是合适于许多需要动态性和易于迭代的场景，但同样的，在许多情况下Python的这些属性恰恰是不利的。后者通常适用的一种环境是要求生产-低延迟和严格部署。
TorchScript可以视为PyTorch模型的一种中间表示，TorchScript表示的PyTorch模型可以直接在C++中进行读取，往往更适合于生产的部署环境。

TorchScript是一种从Pytorch代码创建可序列化和可优化模型的方法。 导出到Torchscript后，模型就可以在Python和c++中运行了。 
- Trace：输入通过模型发送，所有操作都记录在一个将定义您的torchscript模型的图中。 
- Script：如果您的模型更复杂并且具有诸如条件语句之类的控制流，脚本将检查模型的源代码并将其编译为TorchScript代码。


# 1. Environment 运行环境
```shell
pip install --upgrade git+https://github.com/autodeployai/daas-client.git
```