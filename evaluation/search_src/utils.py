import os
import re
import urllib
import requests
from PIL import Image
from io import BytesIO
import hashlib
import time
import logging
from urllib.parse import urlparse, urlsplit

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(funcName)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


def ensure_directory_exists(directory_path):
    """
    检查文件夹是否存在，如果不存在则创建它。
    
    :param directory_path: 目标文件夹的路径
    """
    directory_path = os.path.dirname(directory_path)

    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"目录 {directory_path} 不存在，已创建。")
    else:
        pass
        # print(f"目录 {directory_path} 已存在。")



def is_english_or_chinese(text):
    # 更新正则表达式，允许更多中文和英文标点符号
    pattern = re.compile(r'^[\u4e00-\u9fffA-Za-z0-9\s\-\'\".,!?。？，！、【】：:《》_|“”‘’/|（）()^&*#$%@¥……〜·–]+$')
    return bool(pattern.match(text))


def base64_to_image(base64_str, output_path):
    """
    Converts a Base64 string to an image file.

    :param base64_str: The Base64 string representing the image.
    :param output_path: The path where the image will be saved.
    """
    # If the Base64 string has a data URI scheme, remove it
    if base64_str.startswith('data:image'):
        header, base64_str = base64_str.split(',', 1)

    try:
        # Decode the Base64 string
        image_data = base64.b64decode(base64_str)
    except base64.binascii.Error as e:
        print("Error decoding Base64 string:", e)
        return False

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    try:
        # Write the binary data to a file
        with open(output_path, 'wb') as image_file:
            image_file.write(image_data)
        return True
        # print(f"Image successfully saved to {output_path}")
    except IOError as e:
        print("Error writing image to file:", e)
        return False
    return False


def encode_url(name_):
    dirname_ = os.path.dirname(name_)
    basename_ = os.path.basename(name_)

    encoded_filename = urllib.parse.quote(basename_)    
    return os.path.join(dirname_, encoded_filename)



def get_extension_from_url(url):
    if '.jpg' in url.lower():
        return '.jpg'
    elif '.jpeg' in url.lower():
        return '.jpeg'
    elif '.png' in url.lower():
        return '.png'
    elif '.img' in url.lower():
        return '.jpg'
    elif '.image' in url.lower():
        return '.jpg'
    else:
        # Parse the URL to extract the path
        parsed_url = urlsplit(url)
        path = parsed_url.path
        
        # Extract the extension from the last part of the path
        ext = os.path.splitext(path)[-1]
        
        # Handle cases where the path might not contain a valid extension
        if not ext:
            ext = '.jpg'  # Default to .jpg if no valid extension is found
    
        return ext

def save_image_from_url(url, save_dir):
    '''
    Returns
    -------
    False, if failed to download
    file_path, if success.
    '''
    try:
        logging.info(f"request.getting {url}")
        response = requests.get(url, timeout=5)
        response.raise_for_status()  # Raises an HTTPError for bad responses

        # Get a filename from the URL
        url_hash = hashlib.md5(url.encode()).hexdigest()
        ext = get_extension_from_url(url)

        # timestamp = int(time.time())
        # filename = f"{url_hash}_{timestamp}{ext}"
        
        filename = f"{url_hash}{ext}"
        file_path = os.path.join(save_dir, filename)

        # Open the image and save
        with Image.open(BytesIO(response.content)) as img:
            img.save(file_path)
        logging.info(f"Saved image from {url} as {filename}")
        return file_path

    except requests.RequestException as e:
        logging.error(f"Failed to retrieve {url}: {e}")
    except IOError as e:
        logging.error(f"Failed to save image from {url}: {e}")
    except Exception as e:
        logging.exception("An unexpected error occurred")
    
    return False

