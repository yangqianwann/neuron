import numpy as np
from PIL import Image

import matplotlib.pyplot as plt
from scipy import ndimage as ndi
 
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from skimage.measure import regionprops


def ious(pred, target, num_class):
    ious = np.zeros((num_class))
    for cls in range(num_class):
        pred_inds = pred == cls
        target_inds = target == cls
        intersection = pred_inds[target_inds].sum()
        union = pred_inds.sum() + target_inds.sum() - intersection
        if union == 0:
            ious[cls]=0
            #ious.append(float('nan'))  # if there is no ground truth, do not include in evaluation
        else:
            #ious.append(float(intersection) / max(union, 1))
            ious[cls]=float(intersection) / max(union, 1)
        # print("cls", cls, pred_inds.sum(), target_inds.sum(), intersection, float(intersection) / max(union, 1))
    return np.mean(ious)

def pixel_accs(pred, target):
    correct = (pred == target).sum()
    total   = (target == target).sum()
    return correct / total

def dice_coefs(pred, target, num_class):
    smooth = 1e-5
    dice_coefs = np.zeros((num_class))
    for cls in range(num_class):
        pred_inds = pred == cls
        target_inds = target == cls
        intersection = pred_inds[target_inds].sum()
        dice_coefa=(2 * intersection + smooth) / (pred_inds.sum() + target_inds.sum() + smooth)
        dice_coefs[cls]=float(dice_coefa)
    return np.mean(dice_coefs)

def find_centroids(img):
    props = regionprops(img)
    cent = []
    for i in range(np.unique(img).shape[0]-1):
        cent.append(props[i].centroid)
    return np.array(cent)

def save_result_comparison(epoch, input_np, input_mask, output_np):
    original_im = np.zeros((256,256))
    original_im[:,:]=input_np[0,0,:,:]
    im_seg = np.zeros((256, 256))
    im_mask = np.zeros((256, 256))
    Path = '/home2/zhangwei/project/output_nestedunet/test/'

    # the following version is designed for 11-class version and could still work if the number of classes is fewer.
    for i in range(256):
        for j in range(256):
            if output_np[i, j] == 0:
                im_seg[i, j] = 0 #255
            elif output_np[i, j] == 1:
                im_seg[i, j] = 255 #0
            if input_mask[0,i,j] == 0:
                im_mask[i, j] = 0 #255
            elif input_mask[0,i,j] == 1:
                im_mask[i, j] = 255 #0

    # horizontally stack original image and its corresponding segmentation results
    hstack_image = np.hstack((original_im, im_seg, im_mask))
    new_im = Image.fromarray(np.uint8(hstack_image))

    output_label = im_seg

    distance = ndi.distance_transform_edt(output_label)
    local_maxi = peak_local_max(distance, indices=False, footprint=np.ones((3, 3)),\
                            labels=output_label)
    markers = ndi.label(local_maxi)[0]
    labels = watershed(-distance, markers, mask=output_label,\
             watershed_line=True)
    # labels_1 is the binary labels.
    labels_1 = np.zeros_like(labels)
    labels_1[labels >= 1] = 255

    hstack_image = np.hstack((original_im, labels_1, im_mask))

    """
    plt.subplots_adjust(wspace=1, hspace=1)
    plt.subplot(231).plot(original_im)
    plt.subplot(232).plot(labels, cmap='nipy_spectral')
    plt.subplot(233).plot(im_mask, cmap='gray')
    """
    cent = find_centroids(labels)



    original_label = im_mask 

    distance_1 = ndi.distance_transform_edt(original_label)
    local_maxi = peak_local_max(distance_1, indices=False, footprint=np.ones((3, 3)),\
                            labels=original_label)
    markers = ndi.label(local_maxi)[0]
    original_lb = watershed(-distance, markers, mask=original_label,\
             watershed_line=True)

    cent_original = find_centroids(original_lb)


    fig, axes = plt.subplots(ncols=4,figsize=(18,6), sharex=True, sharey=True)
    ax = axes.ravel()
    
    ax[0].imshow(original_im, cmap=plt.cm.gray)
    ax[1].imshow(labels, cmap=plt.cm.nipy_spectral)
    ax[1].scatter(cent[:,1], cent[:,0], c='#ff63fa', s=4)
    ax[2].imshow(im_mask, cmap=plt.cm.gray)
    
    for a in ax:
        a.set_axis_off()
    
    plt.savefig(Path + str(epoch) + '.png')
    np.save(Path + str(epoch) + '.npy', cent)
    np.save(Path + str(epoch) + '_o.npy', cent_original)
    #plt.show()
    #output_im = Image.fromarray(np.uint8(output_label))
    #hstack_image = np.hstack((original_im, labels_1, im_mask))
    new_im = Image.fromarray(np.uint8(hstack_image))

    file_name = Path + str(epoch) + '.jpg'
    new_im.save(file_name)
    #output_im.save(file_name)
    print('successfully save result')
