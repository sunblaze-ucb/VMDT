"""Helper functions to use with QwenVL-2.5, as seen here: https://github.com/QwenLM/Qwen2.5-VL/blob/main/cookbooks/spatial_understanding.ipynb as well as the main page of the github repo;"""

import json
import time
from json.decoder import JSONDecodeError
import random
import os
import io
import ast
from PIL import Image, ImageDraw, ImageFont
from PIL import ImageColor
import xml.etree.ElementTree as ET
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
# import joblib
from diskcache import Cache
import hashlib
import asyncio
from openai import AsyncOpenAI

# cache = joblib.Memory(location="cache", verbose=0)

additional_colors = [colorname for (colorname, colorcode) in ImageColor.colormap.items()]

def decode_xml_points(text):
    try:
        root = ET.fromstring(text)
        num_points = (len(root.attrib) - 1) // 2
        points = []
        for i in range(num_points):
            x = root.attrib.get(f'x{i+1}')
            y = root.attrib.get(f'y{i+1}')
            points.append([x, y])
        alt = root.attrib.get('alt')
        phrase = root.text.strip() if root.text else None
        return {
            "points": points,
            "alt": alt,
            "phrase": phrase
        }
    except Exception as e:
        print(e)
        return None

def plot_bounding_boxes(im, bounding_boxes, input_width, input_height):
    """
    Plots bounding boxes on an image with markers for each a name, using PIL, normalized coordinates, and different colors.

    Args:
        img_path: The path to the image file.
        bounding_boxes: A list of bounding boxes containing the name of the object
         and their positions in normalized [y1 x1 y2 x2] format.
    """

    # Load the image
    img = im
    width, height = img.size
    print(img.size)
    # Create a drawing object
    draw = ImageDraw.Draw(img)

    # Define a list of colors
    colors = [
    'red',
    'green',
    'blue',
    'yellow',
    'orange',
    'pink',
    'purple',
    'brown',
    'gray',
    'beige',
    'turquoise',
    'cyan',
    'magenta',
    'lime',
    'navy',
    'maroon',
    'teal',
    'olive',
    'coral',
    'lavender',
    'violet',
    'gold',
    'silver',
    ] + additional_colors

    # Parsing out the markdown fencing
    if isinstance(bounding_boxes, str):
        bounding_boxes = parse_json(bounding_boxes)

    # font = ImageFont.truetype("NotoSansCJK-Regular.ttc", size=14)
    font = ImageFont.load_default()

    if isinstance(bounding_boxes, str):
        try:
          json_output = ast.literal_eval(bounding_boxes)
        except Exception as e:
          end_idx = bounding_boxes.rfind('"}') + len('"}')
          truncated_text = bounding_boxes[:end_idx] + "]"
          json_output = ast.literal_eval(truncated_text)
    else:
        json_output = bounding_boxes

    # Iterate over the bounding boxes
    for i, bounding_box in enumerate(json_output):
      # Select a color from the list
      color = colors[i % len(colors)]

      # Convert normalized coordinates to absolute coordinates
      abs_y1 = int(bounding_box["bbox_2d"][1]/input_height * height)
      abs_x1 = int(bounding_box["bbox_2d"][0]/input_width * width)
      abs_y2 = int(bounding_box["bbox_2d"][3]/input_height * height)
      abs_x2 = int(bounding_box["bbox_2d"][2]/input_width * width)

      if abs_x1 > abs_x2:
        abs_x1, abs_x2 = abs_x2, abs_x1

      if abs_y1 > abs_y2:
        abs_y1, abs_y2 = abs_y2, abs_y1

      # Draw the bounding box
      draw.rectangle(
          ((abs_x1, abs_y1), (abs_x2, abs_y2)), outline=color, width=4
      )

      # Draw the text
      if "label" in bounding_box:
        draw.text((abs_x1 + 8, abs_y1 + 6), bounding_box["label"], fill=color, font=font)

    # Display the image
    # img.show()
    return img


def plot_points(im, text, input_width, input_height):
  img = im
  width, height = img.size
  draw = ImageDraw.Draw(img)
  colors = [
    'red', 'green', 'blue', 'yellow', 'orange', 'pink', 'purple', 'brown', 'gray',
    'beige', 'turquoise', 'cyan', 'magenta', 'lime', 'navy', 'maroon', 'teal',
    'olive', 'coral', 'lavender', 'violet', 'gold', 'silver',
  ] + additional_colors
  xml_text = text.replace('```xml', '')
  xml_text = xml_text.replace('```', '')
  data = decode_xml_points(xml_text)
  if data is None:
    img.show()
    return
  points = data['points']
  description = data['phrase']

  font = ImageFont.truetype("NotoSansCJK-Regular.ttc", size=14)

  for i, point in enumerate(points):
    color = colors[i % len(colors)]
    abs_x1 = int(point[0])/input_width * width
    abs_y1 = int(point[1])/input_height * height
    radius = 2
    draw.ellipse([(abs_x1 - radius, abs_y1 - radius), (abs_x1 + radius, abs_y1 + radius)], fill=color)
    draw.text((abs_x1 + 8, abs_y1 + 6), description, fill=color, font=font)
  
  img.show()
  

# @title Parsing JSON output
def parse_json(json_output):
    # Parsing out the markdown fencing
    lines = json_output.splitlines()
    for i, line in enumerate(lines):
        if line == "```json":
            json_output = "\n".join(lines[i+1:])  # Remove everything before "```json"
            json_output = json_output.split("```")[0]  # Remove everything after the closing "```"
            break  # Exit the loop once "```json" is found
    return json_output

def init_qwenvl(model_path = "Qwen/Qwen2.5-VL-7B-Instruct"):
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2",device_map="auto")
    processor = AutoProcessor.from_pretrained(model_path)
    return model, processor

class AsyncCache:
    def __init__(self, cache_path):
        self._cache = Cache(cache_path)
        self._lock = asyncio.Lock()
    
    async def get(self, key):
        async with self._lock:
            return await asyncio.to_thread(self._cache.get, key)
    
    async def set(self, key, value, expire=None):
        async with self._lock:
            return await asyncio.to_thread(self._cache.set, key, value, expire=expire)

async_cache = AsyncCache("./async_cache")

# added model_name as a parameter so we can change the cache key based on the model
# @cache.cache(ignore=['model', 'processor'])
async def inference_qwenvl(model_name, model, processor, img_urls, prompt, system_prompt="You are a helpful assistant", max_new_tokens=1024, use_vllm=True, response_format=None):
  print(f'{img_urls=}')
  images = [Image.open(img_url) for img_url in img_urls]  # really only need for non vllm but I sometimes write the bbox image.
  if use_vllm:
      # call through vllm openai
      openai_api_key = "EMPTY"
      openai_api_base = "http://localhost:8001/v1"
      client = AsyncOpenAI(api_key=openai_api_key, base_url=openai_api_base)
      # image urls must be from the allowed local media path of /ib-scratch/chenguang02/scratch1/cnicholas, which is set in `cnicholas/vllm/launch_vllm_and_balancer.py` when calling vllm serve. Must give the absolute path to the image.
      # img_urls = [os.path.relpath(img_url, "/ib-scratch/chenguang02/scratch1/cnicholas") for img_url in img_urls]
      img_urls = ["file://" + os.path.abspath(img_url) for img_url in img_urls]
      messages = [
        {
          "role": "system",
          "content": system_prompt
        },
        {
          "role": "user",
          "content": [
            {
              "type": "text",
              "text": prompt
            }] + 
            [{
              "type": "image_url",
              "image_url": {"url": img_url}
            } for img_url in img_urls
          ]
        }
      ]

      kwargs = {
            "model": model_name,
            "messages": messages,
            "temperature": 0.0,
            # "top_p": 0.95,  # I don't think we actually change it when doing it here. I think it's always 1.0, which is fine by me.
            "max_tokens": max_new_tokens
      }
      if response_format:
          response_format = response_format["json_schema"]  # convert from openai format to json schema.
          response_format.pop("strict", None)  # remove strict key as it is not used in json schemas.
          import pdb; pdb.set_trace()
          kwargs["extra_body"] = {"guided_json": response_format}  # special format for vllm: https://docs.vllm.ai/en/stable/features/structured_outputs.html.
      # Create a unique cache key using a hash of the inputs
      cache_key = hashlib.sha256(
          json.dumps({
              "model": model_name,
              "messages": messages,
              "max_tokens": max_new_tokens,
              "max-model-len": "8192" if "Qwen" in model_name else "12288"
          }, sort_keys=True).encode()
      ).hexdigest()

      # For retry below. Must define here for use with cache.
      attempts = 0
      num_attempts = 3
  
      # Check if the result is already in the cache
      cached_response = await async_cache.get(cache_key)
      if cached_response is not None:
          print("Cache hit")
          output_text = cached_response
          attempts = num_attempts  # to skip the while loop below.
      # else:
      #     raise ValueError(f"\nCache miss\n\n{cache_key=}\n\n{messages=}\n\n{img_urls=}\n\n{kwargs=}\n\n")
      #     return

      # add retry for error='Backend timed out. Try another instance.' in openai.
      base_sleep = 2
      # # debug:
      # import pdb; pdb.set_trace()
      while attempts < num_attempts:
          try:
              output_text = await client.chat.completions.create(
                  **kwargs
              )
              # if the attribute error is present, raise an error.
              if hasattr(output_text, "error"):
                  raise TypeError(f"Error: {output_text.error}")
              output_text = output_text.choices[0].message.content
              # Save the result to the cache -- no expiration so it always stays in the cache
              await async_cache.set(cache_key, output_text, expire=None)
              break
          except Exception as e:  # to catch openai error when there is no choices attribute.
              print(f"Error: {e}\n{messages=}\n{output_text=}")
              attempts += 1
              if attempts == num_attempts:
                  print(f"Failed after {num_attempts} attempts.")
                  raise e
                  output_text = ""
              retry_sleep = base_sleep * (2 ** attempts)
              print(f"Retrying in {retry_sleep} seconds.")
              # sleep for a bit before retrying.
              await asyncio.sleep(retry_sleep)
      # output_text = await client.chat.completions.create(
      #       **kwargs
      # )
      # output_text = output_text.choices[0].message.content
      # except TypeError as e:
      #   print(f"Error: {e}\n{messages=}\n{output_text=}")
      #   raise e
  else:
      messages = [
        {
          "role": "system",
          "content": system_prompt
        },
        {
          "role": "user",
          "content": [
            {
              "type": "text",
              "text": prompt
            }] + 
            [{
              "image": img_url
            } for img_url in img_urls
          ]
        }
      ]
      add_vision_id = len(img_urls) > 1
      text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, add_vision_id=add_vision_id)
      # print("input:\n",text)
      inputs = processor(text=[text], images=images if images else None, padding=True, return_tensors="pt").to('cuda')

      output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)  # default to temperature=0.0.
      generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
      output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
      output_text = output_text[0]
      # print("output:\n",output_text[0])

  if response_format:
      try:
          json_output = parse_json(output_text)

          # Do parsing, i.e., remove the newlines. Note this will make the reasoning read worse but is fine bc the model already made its prediction.
          json_output = json_output.replace("\n", "")

          # print(f"pre-parsed output: {json_output=}")
          json_output = json.loads(json_output)
          save_bbox_img = False
          if "description" in json_output and "bounding_boxes" in json_output["description"] and save_bbox_img:
              bbox_dir = os.path.join(os.path.dirname(img_url), "..", "bounding_boxes")
              filename = os.path.basename(img_url).split(".")[0]
              os.makedirs(bbox_dir, exist_ok=True)
              input_height = inputs['image_grid_thw'][0][1]*14
              input_width = inputs['image_grid_thw'][0][2]*14
              plot_bounding_boxes(images[0], json_output["description"]["bounding_boxes"], input_width, input_height).save(os.path.join(bbox_dir, f"{filename}_bbox.png"))
              print(f"Saved image with bounding boxes to {bbox_dir}")
      except Exception as e:
          print(f"Error: {e}")
          print(f"output: {output_text}")
          print(f"json_output: {json_output}")
          # print(f"input_height: {input_height}")
          # print(f"input_width: {input_width}")
          raise JSONDecodeError("Error decoding JSON output", output_text, 0)
  else:
      json_output = output_text
  return json_output  # , input_height, input_width
