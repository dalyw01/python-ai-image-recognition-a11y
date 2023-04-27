from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
from PIL import Image
import torch
import os

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

print('\n+ - - - - - - - - - - - - - - - - - - - - - - - - +')
print('Read Local File In Same Directory')
print('+ - - - - - - - - - - - - - - - - - - - - - - - - +')
print(predict_step(['cat.jpg']))

print('\n+ - - - - - - - - - - - - - - - - - - - - - - - - +')
print('Read Multiple Local Files')
print('+ - - - - - - - - - - - - - - - - - - - - - - - - +')
print(predict_step(['cat.jpg','laurence.jpg','skull.jpg','banana.jpg','chair.jpg']))

print('\n+ - - - - - - - - - - - - - - - - - - - - - - - - +')
print('Read Multiple Files In images Directory')
print('+ - - - - - - - - - - - - - - - - - - - - - - - - +')
# be wary of .DS_Store from the OS loading if you check inside the images directory
for filename in os.listdir('images'):
  print(filename)
  print(predict_step([filename]))
