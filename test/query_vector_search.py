import lancedb
from get_emb import get_text_embedding

uri = "../data/photos-256"
db = lancedb.connect(uri)

image_table = "images"
table = db[image_table]

query_emb = get_text_embedding("A FTC Robotics competition robot")
res = table.search(query_emb).limit(5).nprobes(20).refine_factor(10).to_pandas()
# print the image_id and image_path of each one
print(res[["image_path", "people_ids"]])