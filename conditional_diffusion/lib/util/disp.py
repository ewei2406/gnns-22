import matplotlib.pyplot as plt

def showImages(imgs, size=8):
    ct = len(imgs)
    f, axarr = plt.subplots(1, ct, figsize=(size,size))
    for i in range(ct):
        col = axarr[i].imshow(imgs[i])
        axarr[i].axis('off')