from osgeo import gdal
from helpers import rdconverter 

from owslib.wcs import WebCoverageService

from shapely.geometry import Polygon
import os



class Layer(object):
	"""docstring for LayerBuilder"""
	def __init__(self):
		self.dir_ = os.path.dirname(os.path.realpath(__file__))
		super(Layer, self).__init__()
	
	def extent2polygon(self, extent):
	    """Make a Polygon of the extent of a matplotlib axes"""
	    nw = (extent[0], extent[2])
	    no = (extent[1], extent[2])
	    zo = (extent[1], extent[3])
	    zw = (extent[0], extent[3])
	    polygon = Polygon([nw, no, zo, zw])
	    return polygon

	def get_gdal_dataset(self, x_min, x_max, y_min, y_max, **kwargs):
		"""General interface"""
		return None

class AHNLayer(Layer):
	"""docstring for LayerBuilder"""

	def __init__(self):
		super(AHNLayer, self).__init__()

	def get_gdal_dataset(self, x_min, x_max, y_min, y_max, **kwargs):
		dsm = True
		my_wcs = WebCoverageService(
		    'https://geodata.nationaalgeoregister.nl/ahn3/wcs', version='1.0.0')

		identifier = 'ahn3_05m_dsm' if dsm else 'ahn3_05m_dtm'
		output = my_wcs.getCoverage(identifier=identifier, width=2*round(x_max-x_min), height=2*round(
		    y_max-y_min), bbox=(x_min, y_min, x_max, y_max), format='GEOTIFF_FLOAT32', CRS='EPSG:28992')
		f = open(self.dir_ + '/data/tmp_ahn.tiff', 'wb')
		f.write(output.read())
		f.close()

		raster = gdal.Open(self.dir_ + '/data/tmp_ahn.tiff', gdal.GA_ReadOnly)
		# raster = None
		return raster


if __name__ == '__main__':
	bag = AHNLayer()
	lat = 52.153130
	lng = 4.470210
	x = rdconverter.gps2X(lat, lng)
	y = rdconverter.gps2Y(lat,lng)
	d = 500
	r = bag.get_gdal_dataset(x-d,x+d,y-d,y+d)
	print(r)

