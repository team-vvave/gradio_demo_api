import os
import json
from PIL import Image
import gradio as gr
from gradio_client import Client
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import argparse
from clip_retrieval.clip_client import ClipClient
from googletrans import Translator
import openai
with open("secret.txt", 'r') as f :
    openai.api_key = f.readline().strip()

from func2 import search_by_dialogue
from func3 import get_target_paths, do_retrieve

parser = argparse.ArgumentParser()
parser.add_argument("--host", type=str, default="localhost")
parser.add_argument("--port", type=int, default=7000)
parser.add_argument("--workers", type=int, default=1)
args = parser.parse_args()

CLIP_PORT=13131
INDICE_PATH="index_h14"
IMAGE_DIR = "../dataset_shared2/orig-result"

translator = Translator()
app = FastAPI()

origins = ["*"]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

class TextItem(BaseModel):
    text_kor: str

@app.post('/search-by-text', summary="(clip-retrieval) 텍스트로 이미지 검색",
          description="한국어 문장을 입력하면 영어로 번역된 문장으로 이미지를 검색합니다.")
def api_search_by_query(req_json: TextItem) :
    text_kor = req_json.text_kor
    count = 5

    client = Client(f"http://localhost:{args.port}/demo")
    text_en, search_list = client.predict(api_name="/search_by_query", 
                                          input_text_kor=text_kor, input_count=count)
    return {"text_en" : text_en,
            "search_list" : search_list}

@app.post('/search-by-text-func2', summary="(chatgpt + LaBSE) 텍스트로 이미지 검색",
          description="한국어 문장을 입력하면 이미지를 검색합니다.")
def api_search_by_query_func2(req_json: TextItem) :
    text_kor = req_json.text_kor
    count = 5

    client = Client(f"http://localhost:{args.port}/demo")
    search_list = client.predict(api_name="/search_by_query_func2", 
                                 query=text_kor, count=count)
    return {"search_list" : search_list}




def search_by_query(input_text_kor, input_count) :
    text_en = translator.translate(input_text_kor, src='ko', dest='en').text

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

def parsing_json_for_display(search_list) :
    outputs = []
    for search in search_list :
        img_path = os.path.join(IMAGE_DIR, search['image_path'])
        img = Image.open(img_path).convert('RGB')
        outputs.append(img)
    return outputs

with gr.Blocks() as demo :

    with gr.Tab("text search (clip-retrieval)") :
        with gr.Row() :
            with gr.Column() :
                func1_input_text_kor = gr.Text(label="Input (Kor)", info="한국어로 질문을 입력하세요", value="한 학생이 울고 있는 장면")
                func1_input_count = gr.Slider(label="Max count", info="응답받는 최대 개수를 설정합니다.", minimum=1, maximum=100, step=1, value=10)
                func1_btn_submit = gr.Button(value="Submit", variant='primary')

            with gr.Column() :
                func1_output_text_en = gr.Text(label="Input (En)", info="한국어를 영어로 번역", interactive=False)
                func1_output_list = gr.Json(label="Outpus")
                func1_output_gallery = gr.Gallery(label="Output images", columns=5, interactive=False)

        func1_btn_submit.click(fn=search_by_query,
                        inputs=[func1_input_text_kor, func1_input_count],
                        outputs=[func1_output_text_en, func1_output_list],
                        concurrency_id='default').then(fn=parsing_json_for_display,
                                                        inputs=[func1_output_list],
                                                        outputs=[func1_output_gallery],
                                                        concurrency_id='default')

    with gr.Tab("text search (chatgpt + LaBSE)") :
        with gr.Row() :
            with gr.Column() :
                func2_input_text_kor = gr.Text(label="Input (Kor)", info="한국어로 질문을 입력하세요", value="한 학생이 울고 있는 장면")
                func2_input_count = gr.Slider(label="Max count", info="응답받는 최대 개수를 설정합니다.", minimum=1, maximum=100, step=1, value=5)
                func2_btn_submit = gr.Button(value="Submit", variant='primary')

            with gr.Column() :
                func2_output_list = gr.Json(label="Outpus")
                func2_output_gallery = gr.Gallery(label="Output images", columns=5, interactive=False)

        func2_btn_submit.click(fn=search_query,
                               inputs=[func2_input_text_kor, func2_input_count],
                               outputs=[func2_output_list],
                               concurrency_id='default',
                               api_name='search_by_query_func2').then(fn=parsing_json_for_display,
                                                                      inputs=[func2_output_list],
                                                                      outputs=[func2_output_gallery],
                                                                      concurrency_id='default')
        

    with gr.Tab("text with image search (pic2word)") :
        with gr.Row() :
            with gr.Column() :
                func3_input_text_kor = gr.Text(label="Input (Kor)", info="한국어로 질문을 입력하세요. 캐릭터명) 강효민, 김민정, 김서아, 민아름, 박가을, 이미소, 정해서, 진은설, 차태석, 한유현, 황윤혜",
                                               value="there exists 박가을")
                func3_input_count = gr.Slider(label="Max count", info="응답받는 최대 개수를 설정합니다.", minimum=1, maximum=100, step=1, value=5)
                func3_input_paths = gr.Textbox(label="Target paths", info="대상으로 하는 이미지 경로", lines=10, value=get_target_paths('sample'))
                func3_btn_submit = gr.Button(value="Submit", variant='primary')

            with gr.Column() :
                func3_output_text_en = gr.Text(label="Input (En)", info="한국어를 영어로 번역", interactive=False)
                func3_output_img = gr.Image(label='Character image', interactive=False)
                func3_output_list = gr.Json(label="Outpus")
                func3_output_gallery = gr.Gallery(label="Output images", columns=5, interactive=False)

        func3_btn_submit.click(fn=do_retrieve,
                               inputs=[func3_input_text_kor, func3_input_paths, func3_input_count],
                               outputs=[func3_output_text_en, func3_output_img, func3_output_list],
                               concurrency_id='default').then(fn=parsing_json_for_display,
                                                              inputs=[func3_output_list],
                                                              outputs=[func3_output_gallery],
                                                              concurrency_id='default')

demo.title = "웹툰검색데모"
demo.queue(default_concurrency_limit=1)
# demo.launch(server_name='0.0.0.0', server_port=7000, share=False)

app = gr.mount_gradio_app(app, demo, path='/demo')
uvicorn.run(app, host=args.host, port=args.port, workers=args.workers)