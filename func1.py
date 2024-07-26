import os
from clip_retrieval.clip_client import ClipClient
from googletrans import Translator
translator = Translator()

CLIP_PORT=13131
INDICE_PATH="index_h14"

def translate_kor_to_eng(text_kor) :
    text_en = translator.translate(text_kor, src='ko', dest='en').text
    return text_en

def search_by_text(input_text_kor, input_count) :
    text_en = translate_kor_to_eng(input_text_kor)
    client = ClipClient(url=f"http://localhost:{CLIP_PORT}/knn-service", indice_name=INDICE_PATH,
                        num_images=input_count, deduplicate=False,
                        use_safety_model=False, use_violence_detector=False)

    outputs = client.query(text=text_en)
    results = []
    for output in outputs :
        info = {}

        fn1, fn2 = output['image_path'].split('/')[-2:]
        num, _ = os.path.splitext(fn2)

        info['image_path'] = os.path.join(fn1, fn2)
        info['episode'] = int(fn1)
        info['num'] = int(num)
        info['similarity'] = output['similarity']

        results.append(info)
    
    return text_en, results