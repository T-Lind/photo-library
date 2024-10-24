import lancedb
from get_emb import get_text_embedding

uri = "../data/photos-3"
db = lancedb.connect(uri)

image_table = "images"
table = db[image_table]

query_emb = get_text_embedding("Standing outside at dusk")

print(table.search(query_emb).limit(3).nprobes(20).refine_factor(10).to_pandas())