from clip_retrieval.clip_client import ClipClient
import gradio as gr
import os
from PIL import Image
import uvicorn
import argparse

PORT=13131
INDICE_PATH="index_h14"
IMAGE_DIR = "../dataset_shared2/orig-result"

def translate_kor_to_en(input_text_kor) :
    return input_text_kor

def search_by_query(input_text_en, input_count) :
    client = ClipClient(url=f"http://localhost:{PORT}/knn-service", indice_name=INDICE_PATH,
                        num_images=input_count, deduplicate=False,
                        use_safety_model=False, use_violence_detector=False)

    outputs = client.query(text=input_text_en)
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
    
    return results

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
            input_text_kor = gr.Text(label="Input(Kor)", info="현재는 영어만 됩니다ㅠㅠ", value="A student is crying")
            input_count = gr.Slider(label="Max count", info="응답받는 최대 개수를 설정합니다.", minimum=1, maximum=100, step=1, value=10)
            btn_submit = gr.Button(value="Submit", variant='primary')

        with gr.Column() :
            middle_text_en = gr.Text(label="Input(En)", info="번역 기능이 필요할까요?", interactive=False)
            output_list = gr.Json(label="Outpus")
            output_gallery = gr.Gallery(label="Output images")

    btn_submit.click(fn=translate_kor_to_en,
                     inputs=[input_text_kor],
                     outputs=[middle_text_en],
                     concurrency_id='default'
                     ).then(fn=search_by_query,
                            inputs=[middle_text_en, input_count],
                            outputs=[output_list],
                            concurrency_id='default').then(fn=parsing_json_for_display,
                                                           inputs=[output_list],
                                                           outputs=[output_gallery],
                                                           concurrency_id='default')

demo.title = "웹툰검색데모"
demo.queue(default_concurrency_limit=1)
demo.launch(server_name='0.0.0.0', server_port=7000, share=False)

# 1.215.235.253:17000