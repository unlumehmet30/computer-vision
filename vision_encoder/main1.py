from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
from PIL import Image
import requests
import torch    


model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
processor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
img_url="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRN5M5ygd4QmQlNtFoCTwDSM7AN-uHshiwugg&s"
img=Image.open(requests.get(img_url,stream=True).raw).convert("RGB")
pixel_values=processor(images=img, return_tensors="pt").pixel_values
pixel_values=pixel_values.to(device)    


caption=model.generate(pixel_values=pixel_values, max_length=16)
caption=tokenizer.decode(caption[0], skip_special_tokens=True)
print("Generated Caption:", caption) #generated caption:Generated Caption: a man walking down a street with a suitcase 
  