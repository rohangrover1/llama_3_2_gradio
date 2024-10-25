# The code below should be added to a util.py file

import requests
import json
import base64
from dotenv import load_dotenv, find_dotenv
import os
from wolframalpha import Client
from pygments import highlight, lexers, formatters
import re

def llama32_chatbot(together_api_key, local_image, prompt):
    
    if together_api_key==None or len(together_api_key)==0:
       return "please provide valid together API key"

    if local_image!=None:
      base64_image = encode_image(local_image)
      messages = [
        {"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url",
          "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
        }
        ]
        }        
      ]
      result = llama32(messages, together_api_key)
      return result
    else:
      messages = [
        {"role": "user", "content": [
            {"type": "text", "text": prompt}            
        ]
        }        
      ]
      result = llama32(messages, together_api_key)
      return result

def load_env():
    _ = load_dotenv(find_dotenv())

  # The right API to pass in a prompt (of type string) is the completions API https://docs.together.ai/reference/completions-1
  # The right API to pass in a messages (of type of list of message) is The chat completions API https://docs.together.ai/reference/chat-completions-1

def llama32repi(question, image_url, result, new_question, model_size=11):
    messages = [
      {"role": "user", "content": [
          {"type": "text", "text": question},
          {"type": "image_url", "image_url": {"url": image_url}}
      ]},
      {"role": "assistant", "content": result},
      {"role": "user", "content": new_question}
    ]
    result = llama32(messages, model_size)
    return result

def llama32pi(prompt, image_url, model_size=11):
  messages = [
    {
      "role": "user",
      "content": [
        {"type": "text",
          "text": prompt},
        {"type": "image_url",
          "image_url": {
            "url": image_url}
        }
      ]
    },
  ]
  result = llama32(messages, model_size)
  return result

def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')
  
def llama32(messages, together_api_key, model_size=11):
  load_env()
  model = f"meta-llama/Llama-3.2-{model_size}B-Vision-Instruct-Turbo"
  url = f"{os.getenv('DLAI_TOGETHER_API_BASE', 'https://api.together.xyz')}/v1/chat/completions"
  payload = {
    "model": model,
    "max_tokens": 4096,
    "temperature": 0.0,
    "stop": ["<|eot_id|>","<|eom_id|>"],
    "messages": messages
  }

  # headers = {
  #   "Accept": "application/json",
  #   "Content-Type": "application/json",
  #   "Authorization": f"Bearer {os.getenv('TOGETHER_API_KEY')}"
  # }
  headers = {
    "Accept": "application/json",
    "Content-Type": "application/json",
    "Authorization": f"Bearer {together_api_key}"
  }

  res = json.loads(requests.request("POST", url, headers=headers, data=json.dumps(payload)).content)

  if 'error' in res:
    raise Exception(res['error'])

  return res['choices'][0]['message']['content']

def get_wolfram_alpha_api_key():
    load_env()
    wolfram_alpha_api_key = os.getenv("WOLFRAM_ALPHA_KEY")
    return wolfram_alpha_api_key

def get_tavily_api_key():
    load_env()
    tavily_api_key = os.getenv("TAVILY_API_KEY")
    return tavily_api_key


def llama31(prompt_or_messages, model_size=8, temperature=0, raw=False, debug=False):
    load_env()
    model = f"meta-llama/Meta-Llama-3.1-{model_size}B-Instruct-Turbo"
    print(f"using {type(prompt_or_messages)} format for llama31")
    if isinstance(prompt_or_messages, str):
        prompt = prompt_or_messages
        url = f"{os.getenv('DLAI_TOGETHER_API_BASE', 'https://api.together.xyz')}/v1/completions"
        payload = {
            "model": model,
            "temperature": temperature,
            "prompt": prompt
        }
    else:
        messages = prompt_or_messages
        url = f"{os.getenv('DLAI_TOGETHER_API_BASE', 'https://api.together.xyz')}/v1/chat/completions"
        payload = {
            "model": model,
            "temperature": temperature,
            "messages": messages
        }

    if debug:
        print(payload)

    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.getenv('TOGETHER_API_KEY')}"
    }

    try:
        response = requests.post(
            url, headers=headers, data=json.dumps(payload)
        )
        response.raise_for_status()  # Raises HTTPError for bad responses
        res = response.json()
    except requests.exceptions.RequestException as e:
        raise Exception(f"Request failed: {e}")

    if 'error' in res:
        raise Exception(f"API Error: {res['error']}")

    if raw:
        return res

    if isinstance(prompt_or_messages, str):
        return res['choices'][0].get('text', '')
    else:
        return res['choices'][0].get('message', {}).get('content', '')

import os
import requests
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt

# class to keep in memory the conversation
class Conversation:
    def __init__(self, system=""):
        self.messages = []
        if system:
            self.messages.append({"role": "system", "content": system})
    def generate(self, user_question, model=8, temp=0):
        self.messages.append({"role": "user", "content":user_question})
        response = llama31(self.messages, model, temperature=temp)
        self.messages.append({"role":"assistant", "content":response})
        return response
    

def disp_image(address):
    if address.startswith("http://") or address.startswith("https://"):
        response = requests.get(address)
        img = Image.open(BytesIO(response.content))
    else:
        img = Image.open(address)
    
    plt.imshow(img)
    plt.axis('off')
    plt.show()

def resize_image(img, max_dimension = 1120):
  original_width, original_height = img.size

  if original_width > original_height:
      scaling_factor = max_dimension / original_width
  else:
      scaling_factor = max_dimension / original_height

  new_width = int(original_width * scaling_factor)
  new_height = int(original_height * scaling_factor)

  # Resize the image while maintaining aspect ratio
  resized_img = img.resize((new_width, new_height))

  resized_img.save("images/resized_image.jpg")

  print("Original size:", original_width, "x", original_height)
  print("New size:", new_width, "x", new_height)

  return resized_img


from PIL import Image

def merge_images(image_1, image_2, image_3):
    img1 = Image.open(image_1)
    img2 = Image.open(image_2)
    img3 = Image.open(image_3)
    
    width1, height1 = img1.size
    width2, height2 = img2.size
    width3, height3 = img3.size
    
    print("Image 1 dimensions:", width1, height1)
    print("Image 2 dimensions:", width2, height2)
    print("Image 3 dimensions:", width3, height3)
    
    total_width = width1 + width2 + width3
    max_height = max(height1, height2, height3)
    
    merged_image = Image.new("RGB", (total_width, max_height))
    
    merged_image.paste(img1, (0, 0))
    merged_image.paste(img2, (width1, 0))
    merged_image.paste(img3, (width1 + width2, 0))
    
    merged_image.save("images/merged_image_horizontal.jpg")
    
    print("Merged image dimensions:", merged_image.size)
    return merged_image

# pretty print JSON with syntax highlighting
def cprint(response):
    formatted_json = json.dumps(response, indent=4)
    colorful_json = highlight(formatted_json,
                              lexers.JsonLexer(),
                              formatters.TerminalFormatter())
    print(formatted_json)

# import nest_asyncio
# nest_asyncio.apply()
def wolfram_alpha(query: str) -> str:
    WOLFRAM_ALPHA_KEY = get_wolfram_alpha_api_key()
    client = Client(WOLFRAM_ALPHA_KEY)
    result = client.query(query)

    results = []
    for pod in result.pods:
        if pod["@title"] == "Result" or pod["@title"] == "Results":
          for sub in pod.subpods:
            results.append(sub.plaintext)

    return '\n'.join(results)


def html_tokens(tokens):
  # simulate the color values used in https://tiktokenizer.vercel.app
  on_colors = ["#ADE0FC", "#FCE278", "#B2D1FE", "#AFF7C6", "#FDCE9B", "#97F1FB", "#DEE1E7", "#E3C9FF", "#BBC6FD", "#D1FB8C"]

  # Create an HTML string with colored spans
  html_string = ""
  for i, t in enumerate(tokens):
      if t == "\n":
            t = "\\n"
      elif t == "\n\n":
            t = "\\n\\n"
      on_col = on_colors[i % len(on_colors)]
      html_string += f'<span style="color: black; background-color: {on_col}; padding: 2px;">{t}</span>'

  return html_string


# The code below should be added to a util.py file
def llamaguard3(prompt, debug=False):
  model = "meta-llama/Meta-Llama-Guard-3-8B"
  url = f"{os.getenv('DLAI_TOGETHER_API_BASE', 'https://api.together.xyz')}/v1/completions"
  payload = {
    "model": model,
    "temperature": 0,
    "prompt": prompt,
    "max_tokens": 4096,
  }

  headers = {
    "Accept": "application/json",
    "Content-Type": "application/json",
    "Authorization": "Bearer " + os.environ["TOGETHER_API_KEY"]
  }
  res = json.loads(requests.request("POST", url, headers=headers, data=json.dumps(payload)).content)

  if 'error' in res:
    raise Exception(res['error'])

  if debug:
    print(res)
  return res['choices'][0]['text']



def get_boiling_point(liquid_name, celsius):
  # function body
  return []

def extract_query(llama_str):
  # Use regex to extract the sub-string in quotes
  match = re.search(r'query="([^"]+)"', llama_str)
  if match:
      result = match.group(1)
      return result
  else:
      print("No match found")
      return ""


def trending_songs(country_name, top_number):
  try:
      top_number = int(top_number)
  except Exception:
      country_name, top_number = top_number, int(country_name)
     
  songs = {
        "US": [
            "Blinding Lights - The Weeknd",
            "Levitating - Dua Lipa",
            "Peaches - Justin Bieber",
            "Save Your Tears - The Weeknd",
            "Good 4 U - Olivia Rodrigo",
            "Montero (Call Me By Your Name) - Lil Nas X",
            "Kiss Me More - Doja Cat",
            "Stay - The Kid LAROI, Justin Bieber",
            "Drivers License - Olivia Rodrigo",
            "Butter - BTS"
        ],
        "France": [
            "Dernière danse - Indila",
            "Je te promets - Johnny Hallyday",
            "La Vie en rose - Édith Piaf",
            "Tout oublier - Angèle",
            "Rien de tout ça - Amel Bent",
            "J'ai demandé à la lune - Indochine",
            "Bella - Maître Gims",
            "À nos souvenirs - Tino Rossi",
            "Le Sud - Nino Ferrer",
            "La Nuit je mens - Alain Bashung"
        ],
        "Spain": [
            "Despacito - Luis Fonsi",
            "Bailando - Enrique Iglesias",
            "Con altura - Rosalía, J.Balvin",
            "Súbeme la Radio - Enrique Iglesias",
            "Hawái - Maluma",
            "RITMO (Bad Boys for Life) - Black Eyed Peas, J Balvin",
            "Dákiti - Bad Bunny, Jhay Cortez",
            "Vivir mi vida - Marc Anthony",
            "Una vaina loca - Farruko, Sharlene",
            "Te boté - Nio García, Casper Mágico, Ozuna"
        ]
    }

  # Find the list of songs for the given country
  if country_name in songs:
    return songs[country_name][:top_number]

  # If the country is not found, return an empty list
  return []

