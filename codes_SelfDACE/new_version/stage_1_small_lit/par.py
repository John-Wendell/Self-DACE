from thop import profile
import model
import torch
input = torch.randn(1, 3, 1200, 900) #模型输入的形状,batch_size=1
flops, params = profile(model.light_net(), inputs=(input, ))
# print(flops/1e9,params/1e6) #flops单位G，para单位M
print('params:',params/1e6)
print('flops:',flops/1e9)
