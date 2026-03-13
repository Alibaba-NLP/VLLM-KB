# V0.1 初步完成dash接入
操作步骤：
1. apikey请自行通过阿里云申请个人账号（！和公司无关！），只要保证不欠费就OK。
  - [申请apikey](https://help.aliyun.com/zh/model-studio/getting-started/first-api-call-to-qwen)
2. 申请开通和使用模型（默认申请模型是1qps的限速）
  - [模型列表](https://alidocs.dingtalk.com/i/nodes/dQPGYqjpJYZnRbNYCL2GNzno8akx1Z5N) 看第一列，可以一次申请多个模型，英文逗号分割或换行分割，具体可见表单的预填说明。qps是每秒处理请求数，tpm是每分钟处理的token数。通常不紧急任务qps1，500k的tpm就够用了，具体需求自行计算。
  - [申请使用模型](https://yida-group.alibaba-inc.com/APP_CP3FS4XAYKY5WC2ZGLIP/workbench/FORM-5956A3E50C084A0AA46BB10269AD64FA2HW1) 
  - [申请模型quota](https://yida-group.alibaba-inc.com/APP_CP3FS4XAYKY5WC2ZGLIP/workbench/FORM-85680F88B3B44F4790AAEDDAA1F18CEFYJKV)
3. 使用api
  - 参考openai_style_api_client.py进行调用 
  - 参考parallel_run.py做并发调用
  - [错误码](https://help.aliyun.com/zh/model-studio/developer-reference/error-code)