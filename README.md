# python-ai-recognition-script

A script I made to analyze images and provide descriptions

This is something that could be used for web accessiblity in automatically generating alt-text for images

## Basic Setup

If you are familiar with Python and know what you're doing

- First go to [Hugging Face](https://huggingface.co/docs/transformers/installation)
- Once that's setup go to [this project](https://huggingface.co/nlpconnect/vit-gpt2-image-captioning)

Here's some examples

![Picture of a cat with a white background](/images/cat.jpg)

Gives the following description

```
cat.jpg
['a cat sitting on top of a white surface']
```

![Picture of an old woman in batman gear](/images/bat_grandmother.jpg)

```
bat_grandmother.jpg
['a woman wearing a mask and holding a stuffed animal']
```

![Picture of an old woman in batman gear](/images/baby_penguin.jpg)

```
baby_penguin.jpg
['a white and black penguin standing in the snow']
```

![Picture of a painted skull](/images/skull.jpg)

```
skull.jpg
['a painting of a person with a face painted on it']
```

![Picture of a bundle of bananas](/images/banana.jpg)

```
banana.jpg
['bananas sitting on top of each other']
```


