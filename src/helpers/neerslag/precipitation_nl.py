import h5py
import numpy as np
from pyproj import CRS, Transformer

import time
from datetime import timedelta, datetime
import queue
import os

class PrecipitationNL(object):
	"""docstring for PrecipitationNL"""
	def __init__(self, queue_size=36):
		super(PrecipitationNL, self).__init__()
		self.transformer = None
		self.cache = {}
		self.cache_queue = queue.Queue(maxsize=queue_size)
		dire = os.path.dirname(__file__)
		if dire:
			dire += '/'

		is_dslab = os.getenv('DS_LAB', None)
		if is_dslab:
			self.dir = '/local/s2656566/regen_project/src/helpers/neerslag/'
		else:
			self.dir = dire
		
	def get_file_name(self, year, month, day, hour, minute):
		# hours and minutes are 0 index e.g. hours go from 0 to 23

		# floor to closest 5 min
		minute = int(minute/5) * 5

		return self.dir + "data/{}/{:02d}/RAD_NL25_RAC_MFBS_5min_{}{:02d}{:02d}{:02d}{:02d}_NL.h5".format(
				year, month, year, month, day, hour, minute)

	# @helper.timing
	def reproject(self, lat, lon):
		if self.transformer == None:
			proj4_to = "+proj=stere +lat_0=90 +lon_0=0.0 +lat_ts=60.0 +a=6378.137 +b=6356.752 +x_0=0 +y_0=3650"
			proj4_from = "+proj=latlong +datum=WGS84 +R=+12756274"
			# proj_to = wradlib.georef.proj4_to_osr(proj4_to)
			# proj_from = wradlib.georef.proj4_to_osr(proj4_from)
			# result = wradlib.georef.reproject((lon, lat), projection_source=proj_from, projection_target=proj_to)
			# result[1] *= -1

			csr_to = CRS.from_proj4(proj4_to)
			csr_from = CRS.from_proj4(proj4_from)
			self.transformer =  Transformer.from_crs(csr_from, csr_to)
		

		result = self.transformer.transform(lon, lat)

		return result[0], result[1]*-1

	# @helper.timing
	def load_file(self, year, month, day, hour, minute):
		file_name = self.get_file_name(year, month, day, hour, minute)
		if file_name in self.cache:
			file = self.cache[file_name]
		else:
			# we load the file

			if self.cache_queue.full(): #if cache is full we remove the oldest file
				to_remove = self.cache_queue.get()
				del self.cache[to_remove]

			file = h5py.File(file_name)['image1/image_data'][:]
			self.cache[file_name] = file
			self.cache_queue.put(file_name)

		return file


	# @helper.timing
	def get_precipation_data(self, year, month, day, hour, minute, lat, lon):

		coords = self.reproject(lat, lon)
		coords = (int(coords[0]), int(coords[1]))
		data = self.load_file(year, month, day, hour, minute)
		try:
			return data[coords[1],coords[0]]
		except Exception as e:
			return np.nan
		

	# @helper.timing
	def get_precipation_data_past_hours_list(self, year, month, day, hour, minute, lat, lon, n):

		m = int(minute/5) * 5
		date = datetime(year=year, month=month, day=day, hour=hour, minute=minute)
		delta = timedelta(minutes=5)
		_sum = 0
		lst = []
		for i in range(12*n):
			rain = self.get_precipation_data(date.year, date.month, date.day, date.hour, date.minute, lat, lon)
			if rain != 65535:
				_sum += rain
			date = date-delta
			lst.append(rain)

		return lst

	def get_precipation_data_past_hours(self, year, month, day, hour, minute, lat, lon, n):

		m = int(minute/5) * 5
		date = datetime(year=year, month=month, day=day, hour=hour, minute=minute)
		delta = timedelta(minutes=5)
		_sum = 0
		for i in range(12*n):
			rain = self.get_precipation_data(date.year, date.month, date.day, date.hour, date.minute, lat, lon)
			if rain != 65535:
				_sum += rain
			date = date-delta

		return _sum



if __name__ == '__main__':
	pnl = PrecipitationNL()

	d = pnl.get_precipation_data_past_hours_list(2022, 1, 20, 12, 5, 52.1613, 4.49063, 5)
	print(d)













