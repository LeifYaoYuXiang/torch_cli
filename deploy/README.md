# 0. Description 项目介绍
PyTorch的主要接口是Python编程语言。尽管Python是合适于许多需要动态性和易于迭代的场景，但同样的，在许多情况下Python的这些属性恰恰是不利的。后者通常适用的一种环境是要求生产-低延迟和严格部署。
TorchScript可以视为PyTorch模型的一种中间表示，TorchScript表示的PyTorch模型可以直接在C++中进行读取，往往更适合于生产的部署环境。


有两种将PyTorch模型转换为Torch脚本的方法。
第一种称为跟踪，一种机制，其中通过使用示例输入对模型的结构进行一次评估，并记录这些输入在模型中的流量，从而捕获模型的结构。这适用于有限使用控制流的模型。第二种方法是在模型中添加显式批注，以告知Torch Script编
译器可以根据Torch Script语言施加的约束直接解析和编译模型代码。



TorchScript是一种从Pytorch代码创建可序列化和可优化模型的方法。 导出到Torchscript后，模型就可以在Python和c++中运行了。 
- Trace：输入通过模型发送，所有操作都记录在一个将定义您的torchscript模型的图中。 
- Script：如果您的模型更复杂并且具有诸如条件语句之类的控制流，脚本将检查模型的源代码并将其编译为TorchScript代码。





# 1. Environment 运行环境
```shell
pip install --upgrade git+https://github.com/autodeployai/daas-client.git
```