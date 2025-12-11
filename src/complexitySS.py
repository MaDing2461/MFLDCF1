import torch
import time
from thop import profile
from res_unet_plus import ResUnetPlusPlus
import torch.optim as optim

model = ResUnetPlusPlus(3).cuda()
input = torch.randn(1, 3, 256, 256).cuda()
flops, params = profile(model, inputs=(input, ))
print('Complexity: ', flops/1000000000, end=' GFLOPs\n')
print('Parameters: ', params/1000000, end=' M\n')

torch.cuda.synchronize()
time_start = time.time()
predict = model(input)
torch.cuda.synchronize()
time_end = time.time()
print('Speed: ', 1/(time_end-time_start), end=' FPS\n')

# optimizer = optim.SGD(model.parameters(), lr=0.9, momentum=0.9, weight_decay=0.0005)
# for _ in range(1000):
#     optimizer.zero_grad()
#     model(input)
