from datetime import datetime
from haversine import haversine


def timestamp_handle(date):
    date = datetime.fromisoformat(date.replace("/", "-")).timestamp()
    return int(date)


def timestamp2datestr(date):
    date = datetime.fromtimestamp(date)
    return date


def distance(latitude1, longitude1, latitude2, longitude2):
    location1 = (longitude1, latitude1)
    location2 = (longitude2, latitude2)
    dis = haversine(location1, location2)
    return dis