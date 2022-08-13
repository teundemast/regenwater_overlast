from osgeo import gdal

import os


def to_raster(shp, base, name):
    i = 1
    geot = base.GetGeoTransform()
    drv_tiff = gdal.GetDriverByName("GTiff")
    dir_ = os.path.dirname(os.path.realpath(__file__))
    dataset = drv_tiff.Create(dir_ + '/data/{}_tmp.tif'.format(name), base .RasterXSize, base .RasterYSize, 1, gdal.GDT_Byte)
    dataset.SetGeoTransform(geot)
    lyr = shp.GetLayer(0)
    dataset.GetRasterBand(i).SetNoDataValue(0)
    gdal.RasterizeLayer(dataset, [i], lyr, burn_values=[255])
    return dataset.ReadAsArray()




if __name__ == '__main__':
    from layerbuilder.ahn_layer import AHNLayer
    from layerbuilder.bgt_layer import BGTLayer
    from layerbuilder.bag_layer import BAGLayer
     
    x,y = 93659, 463943
    d = 100
    b = (x-d, y-d, x+d, y+d)
    ahn = AHNLayer()
    bag = BAGLayer()
    bgt = BGTLayer()
    ahn_data = ahn.get_gdal_dataset(x-d, x+d, y-d, y+d)
    bag_data = bag.get_gdal_dataset(x-d, x+d, y-d, y+d, intersection_type='single')
    bgt_data = bgt.get_gdal_dataset(x-d, x+d, y-d, y+d, layer='water')

    print(to_raster(bgt_data, ahn_data, 'test'))
    print(to_raster(bag_data, ahn_data, 'test'))
    








