import numpy as np
import matplotlib.pyplot as plt

def get_distance(x, y):
  return np.sqrt((x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2)

def find_nearest(original, new):
  index = np.zeros(original.shape[0])
  for i in range(original.shape[0]):
    d = 100000 
    for j in range(new.shape[0]):
      dis = get_distance(original[i,:], new[j,:])
      if dis < d:
        d = dis
        index[i] = j
  index = index.astype(np.int32)
  return index
  
len = 20 
mean_dis = []
for k in range(len):
	ori_label = np.load(str(k) + '_o.npy')
	new_label = np.load(str(k) + '.npy')

	print(ori_label.shape, new_label.shape)
	#print(ori_label)

	index = find_nearest(ori_label, new_label)
	#print(index[3])

	distance = np.zeros(ori_label.shape[0])
	for i in range(ori_label.shape[0]):
	  distance[i] = get_distance(ori_label[i,:], new_label[index[i], :])

	mean_dis.append(np.sum(distance) / distance.shape)
	
	plt.scatter(ori_label[:, 1],256 - ori_label[:, 0], color = 'blue', alpha=0.8, label='original')
	plt.scatter(new_label[:, 1],256 - new_label[:, 0], color = 'red', alpha=0.85, label = 'new')
	plt.xlabel('image size (256)')
	plt.ylabel('image size (256)')
	plt.title('Centroids')
	plt.legend(loc='upper right')
	plt.savefig(str(k) + '_centroids.png')
	plt.close()

x = np.arange(0,20,1)

plt.scatter(x, mean_dis)
plt.xticks([0,4,8,12,16, 20])
plt.xlabel('sample')
plt.ylabel('mean distance')
plt.title('distances betweens centroids')

plt.savefig('distances.png')
#plt.show()
#print(mean_dis)