# Code from Thijs Simons is used: https://github.com/SimonsThijs/wateroverlast
import requests


class MapsApi(object):
    BASE_URL = '{}/maps/api'

    def __init__(self, key=None, url="https://maps.googleapis.com"):
        self.domain = self.BASE_URL.format(url)
        if key:
            self.key = key

    def _request(self, method, endpoint, headers=None, **kwargs):
        _headers = {
            'Accept': 'application/json',
            'Content-Type': 'application/json',
        }
        if headers:
            _headers.update(headers)

        kwargs['params'].update({'key': self.key})

        return self._parse(requests.request(method, self.domain + endpoint, headers=_headers, **kwargs))

    def get_coordinate(self, address):
        params = {'address': address}
        return self._get('/geocode/json', params=params)

    def _get(self, endpoint, **kwargs):
        return self._request('GET', endpoint, **kwargs)

    def _parse(self, response):
        if 'application/json' in response.headers['Content-Type']:
            r = response.json()
        else:
            return response.text

        return r
