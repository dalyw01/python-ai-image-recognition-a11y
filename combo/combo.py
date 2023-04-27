from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
from PIL import Image
from bs4 import BeautifulSoup
import torch
import os
import requests
import random

# Set the URL of the website you want to download images from
url = "https://github.com/dalyw01/python-ai-image-recognition-a11y"

# Make a GET request to the website and store the HTML content
response = requests.get(url)

# Parse the HTML content using Beautiful Soup
soup = BeautifulSoup(response.content, 'html.parser')

# Find all image tags in the HTML content
img_tags = soup.find_all('img')

# Loop through each image tag
for img_tag in img_tags:
    
  # Get the URL of the image
  img_path = img_tag['src']

  # If its not a Github decorative image
  if img_path.startswith('/dalyw01/python-ai-image-recognition-a11y/'):
    print('+ - - - - - - - - - - - - - - - - - - - - - - - - +')
    print('>' + img_path + '<')

    r = requests.get('https://github.com/' + img_path, stream=True)
    r.raise_for_status()
    r.raw.decode_content = True  

    with Image.open(r.raw) as img:
      # img.show()
      img = img.convert('RGB')
      img.save('images/'+str(random.randrange(1, 1000))+str(random.randrange(1, 1000)) + '.jpg')
      r.close()

model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

max_length = 16
num_beams = 4
gen_kwargs = {"max_length": max_length, "num_beams": num_beams}

def predict_step(image_paths):
  images = []
  for image_path in image_paths:
    i_image = Image.open(image_path)
    if i_image.mode != "RGB":
      i_image = i_image.convert(mode="RGB")

    images.append(i_image)

  pixel_values = feature_extractor(images=images, return_tensors="pt").pixel_values
  pixel_values = pixel_values.to(device)

  output_ids = model.generate(pixel_values, **gen_kwargs)

  preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
  preds = [pred.strip() for pred in preds]
  return preds

print(predict_step(['cat.jpg']))

for filename in os.listdir('images'):
  print('images/'+filename)
  print(predict_step(['images/'+filename]))
