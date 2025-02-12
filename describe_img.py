import torch
from transformers import BlipProcessor, BlipForQuestionAnswering
from transformers import AutoModelForCausalLM, AutoProcessor

def process_blip(image):
    model_name = "ybelkada/blip-vqa-base"
    processor = BlipProcessor.from_pretrained(model_name)
    model = BlipForQuestionAnswering.from_pretrained(model_name, torch_dtype=torch.float16).to("cuda")

    question = "What is this a picture of?"
    inputs = processor(image, question, return_tensors="pt").to("cuda", torch.float16)

    out = model.generate(**inputs)
    return processor.decode(out[0], skip_special_tokens=True)


def process_llama(image):
    model_name = "DAMO-NLP-SG/VideoLLaMA3-2B-Image"
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        device_map="cuda",
        torch_dtype=torch.bfloat16,
        attn_implementation="eager", 
        # VISION_ATTENTION_CLASSES = {
        #     "eager": VisionAttention,
        #     "flash_attention_2": VisionFlashAttention2,  # does not work on my laptop!
        #     "sdpa": VisionSdpaAttention,
        # }
    )

    # Image conversation
    question = "What is this a picture of?"
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question},
            ]
        }
    ]
    inputs = processor(conversation=conversation, return_tensors="pt")
    
    inputs = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
    output_ids = model.generate(**inputs, max_new_tokens=128)
    
    return processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

