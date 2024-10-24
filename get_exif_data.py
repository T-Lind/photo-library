from pillow_heif import register_heif_opener
from PIL import Image
import io

register_heif_opener()

def open_heic_image(file_path):
    with Image.open(file_path) as img:
        with io.BytesIO() as f:
            img.save(f, format='JPEG')
            f.seek(0)
            return Image.open(f)


from exif import Image as ExifImage


def get_timestamp(image_path):
    with open(image_path, 'rb') as image_file:
        img = ExifImage(image_file)

    if 'datetime_original' in img.list_all():
        return img.datetime_original
    return None


# Usage
img_path = r'C:\Users\tenant\PycharmProjects\photo-library\ex-images\IMG_0349.HEIC'
heic_image = open_heic_image(img_path)
timestamp = get_timestamp(img_path)
print(f"Photo taken on: {timestamp}")


def decimal_coords(coords, ref):
    decimal_degrees = coords[0] + coords[1] / 60 + coords[2] / 3600
    if ref == "S" or ref == "W":
        decimal_degrees = -decimal_degrees
    return decimal_degrees


def get_gps_info(image_path):
    with open(image_path, 'rb') as image_file:
        img = ExifImage(image_file)

    if 'gps_latitude' in img.list_all() and 'gps_longitude' in img.list_all():
        lat = decimal_coords(img.gps_latitude, img.gps_latitude_ref)
        lon = decimal_coords(img.gps_longitude, img.gps_longitude_ref)
        return lat, lon
    return None


# Usage
gps_coords = get_gps_info(img_path)
if gps_coords:
    print(f"Latitude: {gps_coords[0]}")
    print(f"Longitude: {gps_coords[1]}")
else:
    print("No GPS coordinates found in image metadata")