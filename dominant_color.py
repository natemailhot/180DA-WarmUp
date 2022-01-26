import cv2 as cv
import numpy as np
cap = cv.VideoCapture(0)
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def find_histogram(clt):
    """
    create a histogram with k clusters
    :param: clt
    :return:hist
    """
    numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
    (hist, _) = np.histogram(clt.labels_, bins=numLabels)

    hist = hist.astype("float")
    hist /= hist.sum()

    return hist

def plot_colors2(hist, centroids):
    bar = np.zeros((50, 300, 3), dtype="uint8")
    startX = 0

    for (percent, color) in zip(hist, centroids):
        # plot the relative percentage of each cluster
        endX = startX + (percent * 300)
        cv.rectangle(bar, (int(startX), 0), (int(endX), 50),
                      color.astype("uint8").tolist(), -1)
        startX = endX

    # return the bar chart
    return bar

clt = KMeans(n_clusters=3) #cluster number

#Loop the video feed
while(1):
    # Take each frame
    _, frame = cap.read()
    
    #Add rectangle frame
    x = 500
    y = 100
    w = 400
    h = 500
    cv.rectangle(frame,(500,100),(900,600),(0,0,255),2)
    crop = frame[500:900, 100:600]   
 
    #img = cv.cvtColor(rect, cv.COLOR_BGR2RGB) 

    img = crop.reshape((crop.shape[0] * crop.shape[1],3)) #represent as row*column,channel number
    clt.fit(img)

    hist = find_histogram(clt)
    bar = plot_colors2(hist, clt.cluster_centers_)

    plt.clf()
    plt.axis("off")
    plt.imshow(bar)
    plt.pause(1)

    # Bitwise-AND mask and original image
    cv.imshow('frame',frame)
    k = cv.waitKey(5) & 0xFF
    if k == 27:
        break
cv.destroyAllWindows()
