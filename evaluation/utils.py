import PIL.Image
from io import BytesIO
import requests
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(funcName)s() - %(filename)s:%(lineno)d - %(message)s'
)

logger = logging.getLogger(__name__)

def _filter_by_url(list_, max_item=5):
    ret = []
    keys_to_check = ['text', 'title', 'snippet']
    # logger.info('Filter image url... It will take some time.')
    for search_data in list_:
        try:
            image_path = search_data.get('image', False) or search_data.get('image_url', False) or search_data.get('image_path', False)
            
            response = requests.get(image_path, timeout=5)
            response.raise_for_status()
            img_data = BytesIO(response.content)
            pil_img = PIL.Image.open(img_data)
            search_data.update({'image': image_path})

            values = [search_data[key] for key in keys_to_check if key in search_data]
            text = "\n".join(values)
            search_data.update({'text': text})

            ret.append(search_data)
        except Exception as e:
            logger.warning(str(e))
            # pass

        if len(ret) == max_item:
            break
    logger.info(f"# Knowledge used: {len(ret)}/{len(list_)}")
    return ret


def _filter_by_text(list_):
    r'''
    Concate values of keys:['text', 'title', 'snippet'] by \n, and name it under 'text'
    '''
    keys_to_check = ['text', 'title', 'snippet']
    ret = []

    for search_data in list_:
        values = [search_data[key] for key in keys_to_check if key in search_data]
        result = "\n".join(values)
        search_data.update({'text': result})
        ret.append(search_data)
    
    logger.info(f"# Knowledge used: {len(ret)}/{len(list_)}")
    return ret


KNOWLEDGE_PREFIX = '''\
Refer to the following knowledge to answer the question. Respond using the same language as the question. \
Knowledge: '''

KNOWLEDGE_PREFIX_NO_CHOICE = '''\
Refer to the following knowledge to answer the question. Do not output A, B, C or D if the question does not ask to. Respond using the same language as the question. \
Knowledge: '''