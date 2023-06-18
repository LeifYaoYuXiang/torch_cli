# 0. Description 项目介绍
在本子项目中，我们将尝试使用Flask来部署PyTorch模型，实现用于模型推断的REST API，到目前为止，以这种方式使用Flask是开始为PyTorch模型提供服务的最简单方法。

在实际的生产环境中，使用Flask结合PyTorch实现REST API不适用于具有高性能要求的用例，可能需要使用TorchScript。

同时，我们利用了第三方库[service-streamer](https://github.com/ShannonAI/service-streamer)库自动将对服务的请求排队，它是一个中间件，将服务请求排队组成一个完整的batch，再送进GPU运算。牺牲最小的时延（默认最大0.1s），提升整体性能，极大提高GPU利用率。


# 1. Environment 运行环境
```shell
pip install flask
pip install service_streamer
```