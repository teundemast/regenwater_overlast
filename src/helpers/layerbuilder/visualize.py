from osgeo import gdal
import matplotlib.pyplot as plt
import numpy as np
import sys
from scipy import ndimage
from scipy.ndimage import gaussian_filter

def showimage(filename):
    filepath = filename
    print(filepath)
    dataset = gdal.Open(filepath, gdal.GA_ReadOnly) # Note GetRasterBand() takes band no. starting from 1 not 0
    width = dataset.RasterXSize
    height = dataset.RasterYSize
    print("width:", width)
    print("height:",height)
    band = dataset.GetRasterBand(1)
    arr = band.ReadAsArray()
    print('dataset')
    print(dataset)
    print('band')
    print(band)
    print('arr')
    print(arr)
    ex = arr.flatten()
    print(ex)
    print('length: '+ str(len(arr)))
    plt.imshow(arr)
    plt.show()
	
def readFile(filename):
    filehandle = gdal.Open(filename)
    band1 = filehandle.GetRasterBand(1)
    geotransform = filehandle.GetGeoTransform()
    geoproj = filehandle.GetProjection()
    band1data = band1.ReadAsArray()
    xsize = filehandle.RasterXSize
    ysize = filehandle.RasterYSize
    print( xsize,ysize)

def sobolFilter(filename):
    filepath = '../../AHN2_5m/' + str(filename)
    print(filepath)
    dataset = gdal.Open(filepath, gdal.GA_ReadOnly) # Note GetRasterBand() takes band no. starting from 1 not 0

    band = dataset.GetRasterBand(1)
    arr = band.ReadAsArray()

    # fig = plt.figure()
    # ax1 = fig.add_subplot(121)  # left side
    # ax2 = fig.add_subplot(122)  # right side

    #fig, axs = plt.subplots(2,2)

    gaussian = gaussian_filter(arr, sigma=5)
    resultG = ndimage.sobel(gaussian)
    result = ndimage.sobel(arr)
    
    fig = plt.figure()
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224)
    ax1.title.set_text('Original')
    ax2.title.set_text('Gaussian')
    ax3.title.set_text('Sobel')
    ax4.title.set_text('Gaussian Sobel')
    

    ax1.imshow(arr)
    ax2.imshow(gaussian)
    ax3.imshow(result)
    ax4.imshow(resultG)
    
    # axs[0,0].imshow(arr)
    # axs[0,0].title.set_text("original")
    # axs[0,1].imshow(gaussian)
    # axs[0,1].title.set_text("gaussian")
    # axs[1,0].imshow(result)
    # axs[1,0].title.set_text("sobel")
    # axs[0,1].imshow(resultG)
    # axs[0,1].title.set_text("gaussian and sobel")
    # ax1.imshow(result)
    # ax2.imshow(resultG)

    
    plt.show()



if __name__ == '__main__':
    #readFile('../../AHN2_5m/' + str(sys.argv[1]))
    showimage("layerbuilder/data/tmp_ahn.tiff")
    #sobolFilter(sys.argv[1])