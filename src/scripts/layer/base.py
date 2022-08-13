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














