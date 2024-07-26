- http://1.215.235.253:17000/demo
- http://1.215.235.253:17000/docs

### Installation
```sh
pip install googletrans-py 
```
- https://github.com/ShivangKakkar/googletrans

### Run clip-retrieval
```sh
clip-retrieval back --port 13131 --indices-paths indices_paths.json --clip_model hf_clip:laion/CLIP-ViT-H-14-laion2B-s32B-b79K --provide_safety_model False --provide_violence_detector False --enable_mclip_option False --provide_aesthetic_embeddings False

```

### Run demo
```sh
python demo.py --host 0.0.0.0 --port 7000 --workers 1
```