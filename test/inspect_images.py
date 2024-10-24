import lancedb
import pandas as pd

uri = "../data/photos-2"
db = lancedb.connect(uri)

# Get the table you want to print
image_table = "images"
table = db[image_table]
# Convert the table to a pandas DataFrame
df = table.to_pandas()

# Print location, timestamp, filename of first 5 timeas
print(df[["image_path", "people_ids"]].head())

people_table = "people"
people = db[people_table]
people_df = people.to_pandas()
print(people_df.head())