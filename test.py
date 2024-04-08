import torch
from torch.utils.tensorboard import SummaryWriter
# from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np

# writer = SummaryWriter('log')
# img_tensor = torch.randn((3, 3200, 1800))
# writer.add_image('test', img_tensor)
# writer.close()

a = [1,2,3]
fig, axs = plt.subplots(nrows=4, ncols=1)
axs[0].plot(a, '-')
axs[0].set_ylabel('LSTM')
axs[1].plot(np.arange(len(a)), a, '-')
axs[2].plot(np.arange(len(a)), a, '-')
axs[3].plot(np.arange(len(a)), a, '-')
# for i in range(2):
#     for j in range(2):
#         axs[i, j].plot([1, 2, 3], [1, 4, 2])
#         axs[i, j].set_title(f'Subplot {i}, {j}')

# plt.show()
plt.savefig('test.png')
