import json
import requests
import time

def google(queries: str, max_retry: int = 10):
    url = "http://101.37.167.147/gw/v1/api/msearch-sp/qwen-search"
    headers = {"Authorization": "Bearer lm-/19WaNVGhRjcjYcKuOV96w==", "Content-Type": "application/json"}
    template = {
        "rid": "test",
        "scene": "dolphin_search_google_nlp",
        "uq": queries[0],
        "debug": False,
        "fields": [],
        "page": 1,
        "rows": 10,
        "customConfigInfo": {
            "multiSearch": False,
            "qpMultiQueryConfig": queries,
            "qpMultiQuery": True,
            "qpMultiQueryHistory": [],
            "qpSpellcheck": False,
            "qpEmbedding": False,
            "knnWithScript": False,
            "rerankSize": 10,
            "qpTermsWeight": False,
            "qpToolPlan": False,
            "inspection": False, #关闭绿网
            "readpage": False,
            "uqLengthLimit": 4000,
            "readpageConfig": {"tokens": 4000, "topK": 10, "onlyCache": False},
        },
        "rankModelInfo": {
            "default": {
                "features": [
                    {"name": "static_value", "field": "_weather_score", "weights": 1.0},
                    {
                        "name": "qwen-rerank",
                        "fields": ["hostname", "title", "snippet", "timestamp_format"],
                        "weights": 1,
                        "threshold": -50,
                        "max_length": 512,
                        "rank_size": 100,
                        "norm": False,
                    },
                ],
                "aggregate_algo": "weight_avg",
            }
        },
        "headers": {
            "__d_head_app": "tomas.wxy",
            "__d_head_qto": 5000
        },
    }

    for _ in range(max_retry):
        try:
            resp = requests.post(url, headers=headers, data=json.dumps(template), timeout=(3, 10))
            rst = json.loads(resp.text)
            docs = rst["data"]["docs"]
            return docs
        except Exception as e:
            print("Meet error when search query:", queries, e)
            print("retrying")
            time.sleep(1 * (_ + 1))
            continue
    return []


if __name__ == "__main__":
    res = google('第二十三届奥运会')
    import pdb
    pdb.set_trace()
    # breakpoint()