import lancedb

uri = "../data/photos-4"
db = lancedb.connect(uri)

# Get the table you want to print
image_table = "images"
imgs_table = db[image_table]
# Convert the table to a pandas DataFrame
imgs_df = imgs_table.to_pandas()
print(imgs_df.shape)
# Print location, timestamp, filename of first 5 timeas
# print(imgs_df[["image_path", "people_ids"]].head(50))

# imgs_table.create_index(num_partitions=16, num_sub_vectors=8)
#
# people_table = "people"
# people = db[people_table]
# people_df = people.to_pandas()
# print(people_df.head())