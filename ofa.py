import os
from functools import partial
import numpy as np
import skimage.transform
import skimage.io
import cv2
import matplotlib.pyplot as plt

#from optical_flow import flow_iterative


def main():

    yosemite = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'data',
        'yosemite_sequence',
        'yos{}.tif'
    )
    fn1 = yosemite.format(2)
    fn2 = yosemite.format(4)

    f1 = skimage.io.imread(fn1).astype(np.double)
    f2 = skimage.io.imread(fn2).astype(np.double)

    # certainties for images - certainty is decreased for pixels near the edge
    # of the image, as recommended by Farneback

    # c1 = np.ones_like(f1)
    # c2 = np.ones_like(f2)

    c1 = np.minimum(1, 1/5*np.minimum(np.arange(f1.shape[0])[:, None], np.arange(f1.shape[1])))
    c1 = np.minimum(c1, 1/5*np.minimum(
        f1.shape[0] - 1 - np.arange(f1.shape[0])[:, None],
        f1.shape[1] - 1 - np.arange(f1.shape[1])
    ))
    c2 = c1
    n_pyr = 4

    opts = dict(
        sigma=4.0,
        sigma_flow=4.0,
        num_iter=3,
        model='constant',
        mu=0,
    )
    d = None

    for pyr1, pyr2, c1_, c2_ in reversed(list(zip(
        *list(map(
            partial(skimage.transform.pyramid_gaussian, max_layer=n_pyr),
            [f1, f2, c1, c2]
        ))
    ))):
        if d is not None:
            d = skimage.transform.pyramid_expand(d, multichannel=True)
            d = d[:pyr1.shape[0], :pyr2.shape[1]]

        d = (pyr1, pyr2)

    xw = d + np.moveaxis(np.indices(f1.shape), 0, -1)
    opts_cv = dict(
        pyr_scale=0.5,
        levels=6,
        winsize=25,
        iterations=10,
        poly_n=25,
        poly_sigma=3.0,
        # flags=0

    )

    d2 = cv2.calcOpticalFlowFarneback(
        f2.astype(np.uint8),
        f1.astype(np.uint8),
        None,
        **opts_cv
    )
    d2 = -d2[..., (1, 0)]

    xw2 = d2 + np.moveaxis(np.indices(f1.shape), 0, -1)

    # ---------------------------------------------------------------
    # use calculated optical flow to warp images
    # ---------------------------------------------------------------

    # opencv warped frame
    f2_w2 = skimage.transform.warp(f2, np.moveaxis(xw2, -1, 0), cval=np.nan)

    # warped frame
    f2_w = skimage.transform.warp(f2, np.moveaxis(xw, -1, 0), cval=np.nan)

    # ---------------------------------------------------------------
    # visualize results
    # ---------------------------------------------------------------

    fig, axes = plt.subplots(2, 2, sharex=True, sharey=True)

    p = 2.0  # percentile of histogram edges to chop off
    vmin, vmax = np.nanpercentile(f1 - f2, [p, 100 - p])
    cmap = 'gray'

    axes[0, 0].imshow(f1, cmap=cmap)
    axes[0, 0].set_title('f1 (fixed image)')
    axes[0, 1].imshow(f2, cmap=cmap)
    axes[0, 1].set_title('f2 (moving image)')
    axes[1, 0].imshow(f1 - f2_w2, cmap=cmap, vmin=vmin, vmax=vmax)
    axes[1, 0].set_title('difference f1 - f2 warped: opencv implementation')
    axes[1, 1].imshow(f1 - f2_w, cmap=cmap, vmin=vmin, vmax=vmax)
    axes[1, 1].set_title('difference f1 - f2 warped: this implementation')
#print("-----------------------------------------------------------------------------------------------------------------")
#print("Result comparison of the various traffic prediction models")
print("------------------------------------------------------------------------------------------------------------------")
print("Comparison of individual classifier performance with ensemble model")
print("------------------------------------------------------------------------------------------------------------------")
d = {"SVM": ["78.34",   "77.23", '76.45'],
#print("------------------------------------------------------------------------------------------------------------")
 "KNN": ["81.23", "80.32", '79.34'],
"Decision Tree": ["93.91", "92.12 ", '91.2'],
"Deep LSTM": ["94.35", "93.43", '92.20'],
"Proposed Ensemble model": ["99.98", "99.80", '99.90'],}
#"GCNN-GRU [38]": ["-        20.88    40.34", "-         -         -", '-          33.12      62.19'],
#"TCC-LSTM-LSM [39]": ["-        -        -", "11.5      12.93     12.92", '16.62      17.12      18.06'] ,
#"AT-Conv-LSTM [40]": ["13.49    14.34    15.48", "10.1      10.8      11.4", '37.39      20.08      18.062326'],

#"WKNN-FDCNN": ["  ", "0.92      0.94      0.96", '12.21      14.23      15.21']}
print("------------------------------------------------------------------------------------------------------------")
print ("{:<30} {:<30} {:<30} {:<10}".format('Method','Accuracy','Sensitivity','Specificity',"\n"))
print("------------------------------------------------------------------------------------------------------------")
for k, v in d.items():
    lang, perc, change = v
    print ("{:<30} {:<30} {:<30} {:<10}".format(k, lang, perc, change))
