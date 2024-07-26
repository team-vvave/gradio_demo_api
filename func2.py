import os
import json
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/LaBSE")


def read_json_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data


def load_scene_data(data_path):
    list_of_documents = []
    data = read_json_file(data_path)
    for datum in data:
        webtoon_num = int(datum['img_path'].split('/')[1])
        cut_num = int(datum['img_path'].split('/')[2].replace('.jpg', ''))
        desc = datum['desc']
        if "###" in desc:
            desc = desc.split("\n###")[0].strip()
        document = Document(page_content=desc, metadata=dict(webtoon_num=webtoon_num, cut_num=cut_num))
        list_of_documents.append(document)
    return list_of_documents


def load_dialogue_data(data_path):
    list_of_documents = []
    data = read_json_file(data_path)
    for datum in data:
        webtoon_num = int(datum['img_path'].split('/')[1])
        cut_num = int(datum['img_path'].split('/')[2].replace('.jpg', ''))
        dialogues = datum['dialogue']
        for dialogue in dialogues:
            dialogue_num = dialogue['dialogue_num']
            desc = dialogue['desc']
            document = Document(page_content=desc, metadata=dict(webtoon_num=webtoon_num, cut_num=cut_num, dialogue_num=dialogue_num))
            list_of_documents.append(document)
    return list_of_documents


def load_db(data_type):
    data_path = f"./func2_retrieval/data/{data_type}_data.json"
    db_name = f"./func2_retrieval/faiss_index_{data_type}"
    if not os.path.exists(db_name):
        if data_type == "scene":
            list_of_documents = load_scene_data(data_path)
        if data_type == "dialogue":
            list_of_documents = load_dialogue_data(data_path)
        db = FAISS.from_documents(list_of_documents, embeddings)
        db.save_local(db_name)
    else:
        print("Load pre-saved FAISS index")
        db = FAISS.load_local(db_name, embeddings, allow_dangerous_deserialization=True)
    return db


def search_by_dialogue(query, count):
    db = load_db(data_type="dialogue")
    results_with_scores = db.similarity_search_with_score(query, k=count)
    
    search_result = []
    for doc, score in results_with_scores:
        episode, num = doc.metadata['webtoon_num'], doc.metadata['cut_num']
        image_path = f"{str(episode).zfill(4)}/{str(num).zfill(3)}.jpg"
        temp = {'content': doc.page_content,
                'image_path': image_path,
                'episode': episode,
                'num': num,
                'similarity': score}
        search_result.append(temp)    
        
    return search_result
