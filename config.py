# 本地明文 API 配置文件。
# 这个文件已被 .gitignore 忽略，可以在本机填写真实 key，不要提交。

# strong model / OpenAI-compatible stages:
# profile_samples.py, question_evolution.py, validate_evolved_question.py,
# collect_answers.py, gen_rubric.py
BASE_URL = "https://hanbbq.labpilot.top/v1"
GPT_MODEL = "gpt-5.4"
HIAPI_KEYS_BIG = [
    "在这里填入你的API_KEY"
]

# 如需按阶段拆分 key，可取消下面注释并分别填写。
# PROFILE_API_KEYS = ["在这里填入profile阶段API_KEY"]
# EVOLVE_API_KEYS = ["在这里填入evolution阶段API_KEY"]
# VALIDATION_API_KEYS = ["在这里填入validation阶段API_KEY"]
# ANSWER_API_KEYS = ["在这里填入answer阶段API_KEY"]
# RUBRIC_API_KEYS = ["在这里填入rubric阶段API_KEY"]

# Qwen candidate / judge model.
# 本地 Qwen 服务不需要 API key，保持空字符串即可。
QWEN_BASE_URL = "http://127.0.0.1:18011/v1"
QWEN_API_KEY = ""
QWEN_MODEL = "hjl_Qwen3.6-27B"

# 如需单独配置 judge，可取消下面注释。
# JUDGE_BASE_URL = QWEN_BASE_URL
# JUDGE_API_KEYS = [QWEN_API_KEY]
# JUDGE_MODEL = QWEN_MODEL
