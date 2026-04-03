# PREFIX_MODEL = '''\
# You are an assistant capable of determining whether a multimodal question-answering sample falls out of your knowledge boundary. Below is a multimodal question-answering sample. Please consider the content of the image, the textual question, and your own knowledge to make one of the following judgments:

# A. My knowledge is enough to answer this question
# B. The image falls out of my knowledge boundary
# C. The text question falls out of my knowledge boundary
# D. Both the image and text fall out of my knowledge boundary

# Example Output: 
# C.

# <image>
# {query}

# Your Output: 
# '''

# V2
PREFIX_MODEL = '''\
You are an assistant designed to solve Visual-Question-Answering (VQA) tasks. The following VQA query may involve knowledge-intensive or time-sensitive content, which might exceed your current capabilities. Please evaluate and respond with one of the following options:

A. My existing knowledge is sufficient to answer this question
B. Additional visual information about the image would be helpful
C. Additional contextual information about the text would be helpful
D. Both visual and textual information would be helpful

Example Output: 
C.

<image>
{query}

Your Output: 
'''


PREFIX_RAG = '''\
You are an assistant capable of determining whether a multimodal question-answering task requires the help of search engines. Below is a multimodal question-answering sample. Please consider the content of the image, the textual question, and your own knowledge to make one of the following judgments regarding what is needed to correctly answer this question:

A. Nothing
B. Perform an image search
C. Perform a text search
D. Perform both image and text searches

Example Output: 
C.

<image>
{query}

Your Output: 
'''


IN = '''A.'''

IMAGE_OUT = '''B.'''

QUERY_OUT = '''C.'''

BOTH_OUT = '''D.'''