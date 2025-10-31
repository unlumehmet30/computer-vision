#import libs
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import requests
import torch

processor=BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model=BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

img_url = "https://static4.depositphotos.com/1003697/435/i/450/depositphotos_4359831-stock-photo-happy-family-playing-with-dog.jpg"
img=Image.open(requests.get(img_url,stream=True).raw).convert("RGB")

input=processor(img, return_tensors="pt").to(device)

with torch.no_grad():
    out=model.generate(**input)
caption=processor.decode(out[0], skip_special_tokens=True)
print("Generated Caption:", caption) #Generated Caption: a family walking on the beach with their dog

