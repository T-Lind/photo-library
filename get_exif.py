import io
import re

import exifread
import pyheif
from PIL import Image

# EXIF IFD pointers and tag IDs
EXIF_IFD = 0x8769
GPS_IFD = 0x8825
DATETIME_ORIGINAL = 36867
DATETIME = 306

GPS_LATITUDE_REF = 1
GPS_LATITUDE = 2
GPS_LONGITUDE_REF = 3
GPS_LONGITUDE = 4


def latlng_conversion(dms, ref):
    """Convert EXIF degree/minute/second rationals to signed decimal degrees."""
    degrees = float(dms[0]) + float(dms[1]) / 60.0 + float(dms[2]) / 3600.0
    if ref in ('S', 'W'):
        degrees = -degrees
    return round(degrees, 5)


def get_coordinates(gps_ifd):
    lat = gps_ifd.get(GPS_LATITUDE)
    lat_ref = gps_ifd.get(GPS_LATITUDE_REF)
    lon = gps_ifd.get(GPS_LONGITUDE)
    lon_ref = gps_ifd.get(GPS_LONGITUDE_REF)

    if not (lat and lat_ref and lon and lon_ref):
        return ""

    return (latlng_conversion(lat, lat_ref), latlng_conversion(lon, lon_ref))


def get_exif_data(ifile):
    ifile = str(ifile)
    if re.search(r'\.(jpe?g|bmp|png)$', ifile, re.IGNORECASE):
        with Image.open(ifile) as image:
            exifdata = image.getexif()

        # DateTimeOriginal lives in the Exif sub-IFD; fall back to the
        # base-IFD DateTime tag if it's missing.
        date = exifdata.get_ifd(EXIF_IFD).get(DATETIME_ORIGINAL) or exifdata.get(DATETIME)

        try:
            geo_loc = get_coordinates(exifdata.get_ifd(GPS_IFD))
        except (TypeError, ValueError, ZeroDivisionError):
            geo_loc = ""

        return date, geo_loc
    elif re.search(r'\.hei[cf]$', ifile, re.IGNORECASE):
        heif_file = pyheif.read(ifile)
        for metadata in heif_file.metadata or []:
            if metadata['type'] == 'Exif':
                fstream = io.BytesIO(metadata['data'][6:])
                tags = exifread.process_file(fstream, details=False)
                date = tags.get('EXIF DateTimeOriginal')
                return (str(date) if date else None), ""

        return None, ""
    elif re.search(r'\.(cr2|nef)$', ifile, re.IGNORECASE):
        # Raw files (Canon and Nikon)
        with open(ifile, 'rb') as f:
            tags = exifread.process_file(f, details=False)

        date = tags.get('EXIF DateTimeOriginal')
        return (str(date) if date else None), ""

    else:
        raise ValueError("File type not supported (doesn't seem to be an image)!")
