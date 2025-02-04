### Project Description for README

---

# **Image Captioning using ViT-GPT2**

## **Overview**
This project implements an **Image Captioning** model using a **Vision Transformer (ViT) as an encoder** and **GPT-2 as a decoder**. The model generates textual descriptions for input images by leveraging pre-trained transformers.

## **Features**
- Utilizes **ViT-GPT2 (Vision Transformer + GPT-2)** for image-to-text generation.
- Uses **Hugging Face's Transformers** library for model loading and inference.
- **Supports GPU acceleration** for faster processing.
- Allows customization of caption generation parameters such as beam search.

## **Dataset and Pretrained Model**
- **Pretrained Model:** `nlpconnect/vit-gpt2-image-captioning`
- The model is fine-tuned for **image captioning tasks**.

## **Dependencies**
To run the project, install the following:
```bash
pip install transformers torch torchvision pillow
```

## **Usage**
1. **Load the model and tokenizer**:
    ```python
    from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
    import torch
    from PIL import Image
    
    model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    ```

2. **Set parameters for text generation**:
    ```python
    max_length = 16
    num_beams = 4
    gen_kwargs = {"max_length": max_length, "num_beams": num_beams}
    ```

3. **Load an image and generate captions**:
    ```python
    def generate_caption(image_path):
        image = Image.open(image_path).convert("RGB")
        inputs = feature_extractor(images=image, return_tensors="pt").to(device)
        output_ids = model.generate(**inputs, **gen_kwargs)
        caption = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return caption

    print(generate_caption("sample_image.jpg"))
    ```

## **Future Work**
- Fine-tuning on a **custom dataset** for improved performance.
- Implementing a **web interface** for easier interaction.
- Enhancing model efficiency using **quantization** techniques.
