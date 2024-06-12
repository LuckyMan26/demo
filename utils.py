__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import json
from langchain_text_splitters import RecursiveJsonSplitter
from openai import OpenAI
import openai
from langchain.vectorstores import Chroma

splitter = RecursiveJsonSplitter(max_chunk_size=3000)

client = OpenAI()
def process_separate_chunk(chunk:str)->str:
    completion = client.chat.completions.create(
      model="gpt-3.5-turbo",
     messages=[
    {"role": "system", "content": "You are assistant. From the given profile extract key technologies and domains with which person has already work or want to learn in the future. Use only information given in the input"},
    {"role": "user", "content": f"Profile: {chunk}"}])
    return completion.choices[0].message.content
def merge_chunks(chunks:list[str])->str:

  completion = client.chat.completions.create(
     model="gpt-3.5-turbo",
     messages=[
    {"role": "system", "content": "Extract all mentioned technologies and domains with which person has worked. Result must contain list of technologies & skills and domains where person has worked"},
    {"role": "user", "content": f"List: {chunks}"}])
  return completion.choices[0].message.content

def clear_json(json:str)->str:
  fields_to_remove = ['objectUrn', 'entityUrn', 'profilePictureDisplayImage', 'rootUrl', 'websites', 'flagshipProfileUrl', 'rootActivity','contactInfo','activityUrl','memberBadges']
  for field in fields_to_remove:
    if field in json:
        del json[field]
  if 'posts' in json and 'insightId' in json['posts']:
    del json['posts']['insightId']
  if 'patents' in json and 'url' in json['patents']:
    del json['patents']['url']


  if 'posts' in json:
    for post in json['posts']:
      if 'activityUnion' in post:
        if 'postActivity' in post['activityUnion']:
          if 'contentSummaryUnion' in post['activityUnion']['postActivity']:
            del post['activityUnion']['postActivity']['contentSummaryUnion']
          if 'activityUrl' in post['activityUnion']['postActivity']:
            del post['activityUnion']['postActivity']['activityUrl']
          if 'entityUrn' in post['activityUnion']['postActivity']:
            del post['activityUnion']['postActivity']['entityUrn']
          if 'entityUrn' in post['activityUnion']['postActivity']['socialMetadata']:
            del post['activityUnion']['postActivity']['socialMetadata']['entityUrn']
          if 'threadUrn' in post['activityUnion']['postActivity']['socialMetadata']:
            del post['activityUnion']['postActivity']['socialMetadata']['threadUrn']
  for education in json['educations']:
    if 'eduId' in education:
      del education['eduId']
    if 'schoolUrn' in education:
      del education['schoolUrn']
    if 'school' in education:
      del education['school']
  return json

def process_file(item_path):
    with open(item_path, 'r') as file:
        data = json.load(file)
        data = clear_json(data)
        json_chunks = splitter.split_json(json_data=data, convert_lists=True)
        chunk_res = [process_separate_chunk(chunk) for chunk in json_chunks]
        res = merge_chunks(chunk_res)
    return res, data['fullName']


