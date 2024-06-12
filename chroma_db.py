import chromadb
from sentence_transformers import SentenceTransformer
from chromadb.config import Settings
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings

from utils import process_file

client  = chromadb.PersistentClient(path="db/")
collection = client.create_collection("profile_summarization")
model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")



def process_one_json(file_paths):
    results = []
    names = []
    for i in range(len(file_paths)):
        result, name = process_file(file_paths[i])
        results.append(result)
        names.append(name)
    return results,names
def process_json_in_batches(json_list, batch_size=50):
    for i in range(0, len(json_list), batch_size):
        batch = json_list[i:i + batch_size]
        process_one_json(batch)

def create_db_collection(results,names,collection):
    for i in range(len(results)):
        collection.add(
        ids=[str(i)],
        documents=results[i],
         metadatas = {"full name": names[i]})

def create_db(json_list, batch_size=50):
    counter = 0
    client = chromadb.PersistentClient(path="db/")
    for i in range(0, len(json_list), batch_size):
        batch = json_list[i:i + batch_size]
        results,names = process_one_json(batch)
        collection = client.create_collection(f"profile_summarization_{counter}")
        create_db_collection(results=results,names=names,collection=collection)
        counter+=1
    return counter