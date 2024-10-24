import lancedb
import pandas as pd

uri = "../data/photos-1"
db = lancedb.connect(uri)

# Get the table you want to print
table_name = "images"
table = db[table_name]
# Convert the table to a pandas DataFrame
df = table.to_pandas()

# Print location, timestamp, filename of first 5 timeas
print(df[["location", "date", "image_path"]].head())
