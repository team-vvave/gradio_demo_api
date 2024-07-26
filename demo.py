import os
from PIL import Image
import gradio as gr
from gradio_client import Client
from fastapi import FastAPI
import uvicorn
import argparse
from clip_retrieval.clip_client import ClipClient
from googletrans import Translator

parser = argparse.ArgumentParser()
parser.add_argument("--host", type=str, default="localhost")
parser.add_argument("--port", type=int, default=7000)
parser.add_argument("--workers", type=int, default=1)
args = parser.parse_args()

PORT=13131
INDICE_PATH="index_h14"
IMAGE_DIR = "../dataset_shared2/orig-result"

app = FastAPI()
translator = Translator()

@app.post('/search-by-text', summary="텍스트로 이미지 검색",
          description="한국어 문장을 입력하면 영어로 번역된 문장으로 이미지를 검색합니다.")
def api_search_by_query(text_kor: str, count: int) :
    client = Client(f"http://localhost:{args.port}/demo")
    text_en, search_list = client.predict(api_name="/search_by_query", 
                                 input_text_kor=text_kor, input_count=count)
    return {"text_en" : text_en,
            "search_list" : search_list}

def search_by_query(input_text_kor, input_count) :
    text_en = translator.translate(input_text_kor, src='ko', dest='en').text

    client = ClipClient(url=f"http://localhost:{PORT}/knn-service", indice_name=INDICE_PATH,
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

def parsing_json_for_display(search_list) :
    outputs = []
    for search in search_list :
        img_path = os.path.join(IMAGE_DIR, search['image_path'])
        img = Image.open(img_path).convert('RGB')
        outputs.append(img)
    return outputs

with gr.Blocks() as demo :

    with gr.Row() :

        with gr.Column() :
            input_text_kor = gr.Text(label="Input (Kor)", info="한국어로 질문을 입력하세요", value="한 학생이 울고 있는 장면")
            input_count = gr.Slider(label="Max count", info="응답받는 최대 개수를 설정합니다.", minimum=1, maximum=100, step=1, value=10)
            btn_submit = gr.Button(value="Submit", variant='primary')

        with gr.Column() :
            output_text_en = gr.Text(label="Input (En)", info="한국어를 영어로 번역", interactive=False)
            output_list = gr.Json(label="Outpus")
            output_gallery = gr.Gallery(label="Output images")

    btn_submit.click(fn=search_by_query,
                     inputs=[input_text_kor, input_count],
                     outputs=[output_text_en, output_list],
                     concurrency_id='default').then(fn=parsing_json_for_display,
                                                    inputs=[output_list],
                                                    outputs=[output_gallery],
                                                    concurrency_id='default')

demo.title = "웹툰검색데모"
demo.queue(default_concurrency_limit=1)
# demo.launch(server_name='0.0.0.0', server_port=7000, share=False)

app = gr.mount_gradio_app(app, demo, path='/demo')
uvicorn.run(app, host=args.host, port=args.port, workers=args.workers)