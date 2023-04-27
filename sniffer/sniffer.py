import requests
import random
from bs4 import BeautifulSoup
from PIL import Image

# Run 'pip install bs4' before executing script

# Set the URL of the website you want to download images from
url = "https://github.com/dalyw01/python-ai-image-recognition-a11y"

# Make a GET request to the website and store the HTML content
response = requests.get(url)

# Parse the HTML content using Beautiful Soup
soup = BeautifulSoup(response.content, 'html.parser')

# Find all image tags in the HTML content
img_tags = soup.find_all('img')

print('X X X X X X X X X X X X X X X X X X X X X X X X X X X ')
print('Printing <img> Tags')
print('X X X X X X X X X X X X X X X X X X X X X X X X X X X ')
print(img_tags)
print('X X X X X X X X X X X X X X X X X X X X X X X X X X X ')

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
            img.show()
            img = img.convert('RGB')
            img.save(str(random.randrange(1, 1000))+str(random.randrange(1, 1000)) + '.jpg')
        r.close() 