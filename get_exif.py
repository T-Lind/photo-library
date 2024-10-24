from PIL import Image
import io
import pyheif
import exifread
import re
from PIL.ExifTags import TAGS, GPSTAGS


def latlng_conversion(latlng, ref):
    degrees = latlng[0][0] / latlng[0][1]
    minutes = latlng[1][0] / latlng[1][1] / 60.0
    seconds = latlng[2][0] / latlng[2][1] / 3600.0

    if ref in ['S', 'W']:
        degrees = -degrees
        minutes = -minutes
        seconds = -seconds

    return round(degrees + minutes + seconds, 5)


def get_coordinates(geotags):
    lat = latlng_conversion(geotags['GPSLatitude'], geotags['GPSLatitudeRef'])

    lon = latlng_conversion(geotags['GPSLongitude'], geotags['GPSLongitudeRef'])

    return (lat, lon)


def get_geotagging(exif):
    if not exif:
        raise ValueError("No EXIF metadata found")

    geotagging = {}
    for (idx, tag) in TAGS.items():
        if tag == 'GPSInfo':
            if idx not in exif:
                raise ValueError("No EXIF geotagging found")

            for (key, val) in GPSTAGS.items():
                if key in exif[idx]:
                    geotagging[val] = exif[idx][key]
    return geotagging


def get_exif_data(ifile):
    if re.search(r'jpeg$|bmp$|png$|jpg$', str(ifile), re.IGNORECASE):
        image = Image.open(ifile)
        exifdata = image.getexif()
        geotags = get_geotagging(exifdata)
        if "{1:" in str(exifdata[34853]):
            lat_long = get_coordinates(geotags)
            # geo_loc = get_location(str(lat_long)[1:-1])
            geo_loc = lat_long

        else:
            geo_loc = ""  # No loc data

        return exifdata.get(36867), geo_loc
    elif re.search(r'heic$', str(ifile), re.IGNORECASE):
        # this part of the decision tree processes HEIC files

        heif_file = pyheif.read(str(ifile))
        for metadata in heif_file.metadata:

            if metadata['type'] == 'Exif':
                fstream = io.BytesIO(metadata['data'][6:])

        tags = exifread.process_file(fstream, details=False)

        return str(tags.get('EXIF DateTimeOriginal')), ""
    elif re.search(r'CR2$|NEF$', str(ifile), re.IGNORECASE):
        # this part of the decision tree processes raw files (Cannon and Nikon)
        f = open(ifile, 'rb')

        # Return Exif tags
        tags = exifread.process_file(f, details=False)
        orig_date = tags['EXIF DateTimeOriginal']

        return str(orig_date)[:10], ""

    else:
        raise ValueError("File type not supported (doesn't seem to be an image)!")
