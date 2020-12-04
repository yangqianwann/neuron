import codecs
import numpy as np
import matplotlib.pyplot as plt

PATH = 'C:/Users/wzhan/Desktop'
title = ['loss', 'iou', 'dice_coef', 'pixel_acc', 'val_loss', \
		'val_iou', 'val_dice_coef', 'val_pixel_acc']
a = np.loadtxt("data.txt", delimiter=',',usecols=(2,3,4,5,6,7,8,9), unpack =False)

loss = np.array(a)
x = np.arange(20)
print(type(loss), np.shape(loss))

plt.subplots_adjust(wspace =1, hspace =1)

for i in range(8):
	plt.subplot(2,4,i+1).plot(x, loss[:,i])
	plt.title(title[i])

plt.savefig('./output.png')
plt.show()

