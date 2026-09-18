---
license: apache-2.0
library_name: PaddleOCR
language:
- en
pipeline_tag: image-to-text
tags:
- OCR
- PaddlePaddle
- PaddleOCR
- textline_recognition
---

# devanagari_PP-OCRv5_mobile_rec

## Introduction

devanagari_PP-OCRv5_mobile_rec is one of the PP-OCRv5_rec that are the latest generation text line recognition models developed by PaddleOCR team. It aims to efficiently and accurately support the recognition of Devanagari. The key accuracy metrics are as follow:

| Model | Accuracy (%) |
|-|-|
| devanagari_PP-OCRv5_mobile_rec | 84.96|



**Note**: If any character (including punctuation) in a line was incorrect, the entire line was marked as wrong. This ensures higher accuracy in practical applications.

## Quick Start

### Installation

1. PaddlePaddle

Please refer to the following commands to install PaddlePaddle using pip:

```bash
# for CUDA11.8
python -m pip install paddlepaddle-gpu==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/

# for CUDA12.6
python -m pip install paddlepaddle-gpu==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/

# for CPU
python -m pip install paddlepaddle==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
```

For details about PaddlePaddle installation, please refer to the [PaddlePaddle official website](https://www.paddlepaddle.org.cn/en/install/quick).

2. PaddleOCR

Install the latest version of the PaddleOCR inference package from PyPI:

```bash
python -m pip install paddleocr
```

### Model Usage

You can quickly experience the functionality with a single command:

```bash
paddleocr text_recognition \
    --model_name devanagari_PP-OCRv5_mobile_rec \
    -i https://cdn-uploads.huggingface.co/production/uploads/684ad4f6eb7d8ee8f6a92a3a/dtgfI1BdDM7c9BB0X3-xE.png
```

You can also integrate the model inference of the text recognition module into your project. Before running the following code, please download the sample image to your local machine.

```python
from paddleocr import TextRecognition
model = TextRecognition(model_name="devanagari_PP-OCRv5_mobile_rec")
output = model.predict(input="dtgfI1BdDM7c9BB0X3-xE.png", batch_size=1)
for res in output:
    res.print()
    res.save_to_img(save_path="./output/")
    res.save_to_json(save_path="./output/res.json")
```

After running, the obtained result is as follows:

```json
{'res': {'input_path': '/root/.paddlex/predict_input/dtgfI1BdDM7c9BB0X3-xE.png', 'page_index': None, 'rec_text': 'कख ग घड', 'rec_score': 0.8845340013504028}}
```

The visualized image is as follows:

![image/jpeg](https://cdn-uploads.huggingface.co/production/uploads/684ad4f6eb7d8ee8f6a92a3a/G6Zqw5OG2kSNopsgBJYWb.png)

For details about usage command and descriptions of parameters, please refer to the [Document](https://paddlepaddle.github.io/PaddleOCR/latest/en/version3.x/module_usage/text_recognition.html#iii-quick-start).

### Pipeline Usage

The ability of a single model is limited. But the pipeline consists of several models can provide more capacity to resolve difficult problems in real-world scenarios.

#### PP-OCRv5

The general OCR pipeline is used to solve text recognition tasks by extracting text information from images and outputting it in string format. And there are 5 modules in the pipeline: 
* Document Image Orientation Classification Module (Optional)
* Text Image Unwarping Module (Optional)
* Text Line Orientation Classification Module (Optional)
* Text Detection Module
* Text Recognition Module

Run a single command to quickly experience the OCR pipeline:

```bash
paddleocr ocr -i https://cdn-uploads.huggingface.co/production/uploads/684ad4f6eb7d8ee8f6a92a3a/4565zt8fsAJ2JCVyP2OEa.png \
    --text_recognition_model_name devanagari_PP-OCRv5_mobile_rec \
    --use_doc_orientation_classify False \
    --use_doc_unwarping False \
    --use_textline_orientation True \
    --save_path ./output \
    --device gpu:0 
```

Results are printed to the terminal:

```json
{'res': {'input_path': '/root/.paddlex/predict_input/4565zt8fsAJ2JCVyP2OEa.png', 'page_index': None, 'model_settings': {'use_doc_preprocessor': True, 'use_textline_orientation': False}, 'doc_preprocessor_res': {'input_path': None, 'page_index': None, 'model_settings': {'use_doc_orientation_classify': False, 'use_doc_unwarping': False}, 'angle': -1}, 'dt_polys': array([[[ 172,  201],
        ...,
        [ 172,  290]],

       ...,

       [[ 795, 2010],
        ...,
        [ 795, 2055]]], shape=(42, 4, 2), dtype=int16), 'text_det_params': {'limit_side_len': 64, 'limit_type': 'min', 'thresh': 0.3, 'max_side_limit': 4000, 'box_thresh': 0.6, 'unclip_ratio': 1.5}, 'text_type': 'general', 'textline_orientation_angles': array([-1, ..., -1], shape=(42,)), 'text_rec_score_thresh': 0.0, 'return_word_box': False, 'rec_texts': ['विषय-सूची', 'क्र.सं.', 'विवरण', 'पृष्ठ सं.', '1', 'आरईसी लिमिटेड और आरईसीआईपीएमटी के बारे में', '2-3', '2', 'विदयुत क्षत्र के अधिकारियों के लिए राष्टीय नियमित प्रशिक्षण कार्यक्रम', '4-10', '3', 'विदयुत क्षत्र के अधिकारियों के लिए आरईसी द्वारा प्रायोजित प्रशिक्षण कार्यक्रम', '11-14', 'आरईसीआईपीएमटी/ऑफ-कैंपस में आरईसी के कर्मचारियों के लिए इन-हाउस', '4', 'प्रशिक्षण कार्यक्रम', '15-16', '5', 'अनुकूलित प्रशिकषण कैंपस', '17-18', '6', 'आरईसीआईपीएमटी कैंपस', '19-21', '7', 'अंतरिक फैकल्टी के सदस्य', '22', '8', 'बाहरी फैकल्टी के सदस्य', '23', 'संस्थानों और यूटिलिटीज के साथ समझौता ज़ापन', '9', '24', 'मिशन', 'अपने अनुभव, विशेषज्ञता को साझञा करने और बिजली', 'यूटिलिटिज के प्रबंधकीय कर्मियों को प्रबुद्ध करने के लिए', 'बिजली क्षेत्र के मानव संसाधन विकास के लिए वैश्िक', 'उत्कृष्टता की एक संस्था का निर्माण करना।', 'विज़न', 'बिजली इंजीनियरों/्रबंधकों तक पहुंचना, शिक्षित करना,', 'प्रित करना, पोषण करना, प्रबुद करना और सक्रय करना', 'और उच्च उत्पादकता प्रस्त करने के लिए मानव संसाधनों', 'में गुणवता सुधार के लिए प्रयास करना।'], 'rec_scores': array([0.96054012, ..., 0.96115613], shape=(42,)), 'rec_polys': array([[[ 172,  201],
        ...,
        [ 172,  290]],

       ...,

       [[ 795, 2010],
        ...,
        [ 795, 2055]]], shape=(42, 4, 2), dtype=int16), 'rec_boxes': array([[ 172, ...,  290],
       ...,
       [ 795, ..., 2061]], shape=(42, 4), dtype=int16)}}
```

If save_path is specified, the visualization results will be saved under `save_path`. The visualization output is shown below:

![image/jpeg](https://cdn-uploads.huggingface.co/production/uploads/684ad4f6eb7d8ee8f6a92a3a/yS8pRQnX1uIlgfjaqPXlu.png)

The command-line method is for quick experience. For project integration, also only a few codes are needed as well:

```python
from paddleocr import PaddleOCR  

ocr = PaddleOCR(
    text_recognition_model_name="devanagari_PP-OCRv5_mobile_rec",
    use_doc_orientation_classify=False, # Use use_doc_orientation_classify to enable/disable document orientation classification model
    use_doc_unwarping=False, # Use use_doc_unwarping to enable/disable document unwarping module
    use_textline_orientation=True, # Use use_textline_orientation to enable/disable textline orientation classification model
    device="gpu:0", # Use device to specify GPU for model inference
)
result = ocr.predict("https://cdn-uploads.huggingface.co/production/uploads/684ad4f6eb7d8ee8f6a92a3a/4565zt8fsAJ2JCVyP2OEa.png")  
for res in result:  
    res.print()  
    res.save_to_img("output")  
    res.save_to_json("output")
```

The default model used in pipeline is `PP-OCRv5_server_rec`, so it is needed that specifing to `devanagari_PP-OCRv5_mobile_rec` by argument `text_recognition_model_name`. And you can also use the local model file by argument `text_recognition_model_dir`. For details about usage command and descriptions of parameters, please refer to the [Document](https://paddlepaddle.github.io/PaddleOCR/latest/en/version3.x/pipeline_usage/OCR.html#2-quick-start).

## Links

[PaddleOCR Repo](https://github.com/paddlepaddle/paddleocr)

[PaddleOCR Documentation](https://paddlepaddle.github.io/PaddleOCR/latest/en/index.html)
