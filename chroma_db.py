__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import chromadb
from sentence_transformers import SentenceTransformer
from chromadb.config import Settings
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from utils import process_file


model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")



def process_json_batch(file_paths,client):
    results = []
    names = []
    for i in range(len(file_paths)):
        result, name = process_file(file_paths[i],client)
        results.append(result)
        names.append(name)
    return results,names


def create_db_collection(results,names,collection):
    for i in range(len(results)):
        collection.add(
        ids=[str(i)],
        documents=results[i],
         metadatas = {"full name": names[i]})

def create_db(json_list,client, batch_size=50):
    counter = 0
    for i in range(0, len(json_list), batch_size):
        batch = json_list[i:i + batch_size]
        results,names = process_json_batch(batch,client)
        collection = client.create_collection(f"profile_summarization_{counter}")
        create_db_collection(results=results,names=names,collection=collection)
        counter+=1
    return counter