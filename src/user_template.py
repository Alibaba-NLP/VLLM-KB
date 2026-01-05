USER_TEMPLATE_CN = \
'''你是一个能够在多模态问答场景中决定是否需要进行图片搜索的助手。下面我将提供给你一个多模态问题，包括文本问题和图片链接。\
请你回答：“true”或者“false”，表示回答此多模态问题是否需要执行搜索。\
<extra_14>文本问题：{0}
<img>{1}</img><extra_15>'''

USER_TEMPLATE_CN_SCORE = \
'''你是一个能够在多模态问答场景中决定是否需要进行图片搜索的助手。下面我将提供给你一个多模态问题，包括文本问题和图片链接。\
请回复一个1.0到5.0之间的分数，以表明是否需要执行搜索来正确回答这个多模态问题。

评分时请遵循以下准则：
- 您的评分必须在1.0到5.0之间，其中1.0表示不需要搜索，5.0表示需要搜索。
- 评分不必是整数。
示例回答：
4.0

<extra_14> 文本问题：{0}
<img>{1}</img><extra_15>
您的评分：'''

USER_TEMPLATE_EN = \
'''You are an assistant capable of deciding whether an image search is needed in a multimodal question-answering scenario. Below, I will provide you with a multimodal question that includes a text question and an image link.
Please respond with "true" or "false," indicating whether a search is necessary (true) or not (false) to answer this multimodal question.
<extra_14> Text question: {0}
<img>{1}</img><extra_15>'''

USER_TEMPLATE_EN_SCORE = \
'''You are an assistant capable of deciding whether an image search is needed in a multimodal question-answering scenario. Below, I will provide you with a multimodal question that includes a text question and an image link.
Please respond with a score ranging from 1.0 to 5.0 indicating whether a search is necessary or not to answer this multimodal question. 

Follow these guidelines for scoring:
- Your score has to be between 1.0 and 5.0, where 1.0 stands for an unnecessary search and 5.0 stands for a necessary search. 
- The score does not have to be integer.

Example Response:
4.0

<extra_14> Text question: {0}
<img>{1}</img><extra_15>

Your score: '''



USER_TEMPLATE_EN_DS = \
'''You are an assistant capable of deciding whether an image search is needed in a multimodal question-answering scenario. Below, I will provide you with a multimodal question that includes a text question and an image link.
Please respond with "true" or "false," indicating whether a search is necessary (true) or not (false) to answer this multimodal question.
Question: {0}

Your Response: '''

USER_TEMPLATE_EN_SCORE_DS = \
'''You are an assistant capable of deciding whether an image search is needed in a multimodal question-answering scenario. Below, I will provide you with a multimodal question that includes a text question and an image link.
Please respond with a score ranging from 1.0 to 5.0 indicating whether a search is necessary or not to answer this multimodal question. 

Follow these guidelines for scoring:
- Your score has to be between 1.0 and 5.0, where 1.0 stands for an unnecessary search and 5.0 stands for a necessary search. 
- The score does not have to be integer.

Example Response:
4.0

Question: {0}

Your Response: '''


USER_TEMPLATE_EN_DS_INFERENCE = \
'''You are an assistant capable of deciding whether an image search is needed in a multimodal question-answering scenario. Below, I will provide you with a multimodal question that includes a text question and an image link.
Please respond with "true" or "false," indicating whether a search is necessary (true) or not (false) to answer this multimodal question.

Question: {0}
<img>{1}</img>

Your Response: '''

USER_TEMPLATE_EN_SCORE_DS_INFERENCE = \
'''You are an assistant capable of deciding whether an image search is needed in a multimodal question-answering scenario. Below, I will provide you with a multimodal question that includes a text question and an image link.
Please respond with a score ranging from 1.0 to 5.0 indicating whether a search is necessary or not to answer this multimodal question. 

Follow these guidelines for scoring:
- Your score has to be between 1.0 and 5.0, where 1.0 stands for an unnecessary search and 5.0 stands for a necessary search. 
- The score does not have to be integer.

Example Response:
4.0

Question: {0}
<img>{1}</img>

Your Response: '''


USER_TEMPLATE_CN_DS_INFERENCE = \
'''你是一个能够决定在多模态问答场景中是否需要进行图像搜索的助手。下面，我将提供一个包含文本问题和图像链接的多模态问题。
请用“true”或“false”回答，表示是否需要搜索（true）或不需要搜索（false）来回答这个多模态问题。

问题: {0}
<img>{1}</img>

你的回答: '''

USER_TEMPLATE_CN_SCORE_DS_INFERENCE = \
'''你是一个能够决定在多模态问答场景中是否需要进行图像搜索的助手。下面，我将提供一个包含文本问题和图像链接的多模态问题。
请用1.0到5.0的分数来回答，表示是否需要搜索来回答这个多模态问题。

请遵循以下评分指南：
- 你的评分必须在1.0到5.0之间，其中1.0表示不需要搜索，5.0表示需要搜索。
- 评分不必是整数。

示例回答：
4.0

问题: {0}
<img>{1}</img>

你的回答: '''
