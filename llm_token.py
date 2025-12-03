from llama_cpp import Llama
from openai import OpenAI
from loguru import logger
from time import sleep
import tiktoken  

GLOBAL_LLM = None

# 适配Qwen2.5的Tokenizer（兼容OpenAI/LLaMA系模型）
def get_tokenizer():
    try:
        return tiktoken.encoding_for_model("gpt-3.5-turbo")
    except:
        # 兜底：使用cl100k_base编码（适配绝大多数模型）
        return tiktoken.get_encoding("cl100k_base")

class LLM:
    def __init__(self, api_key: str = None, base_url: str = None, model: str = None, lang: str = "English"):
        self.lang = lang
        self.tokenizer = get_tokenizer()
        self.max_ctx = 5000  # 与n_ctx保持一致
        self.reserved_output_tokens = 200  # 预留200 Token给输出
        
        if api_key:
            self.llm = OpenAI(api_key=api_key, base_url=base_url)
            self.model = model
        else:
            self.llm = Llama.from_pretrained(
                repo_id="Qwen/Qwen2.5-3B-Instruct-GGUF",
                filename="qwen2.5-3b-instruct-q4_k_m.gguf",
                n_ctx=self.max_ctx,  # 显式绑定上下文窗口
                n_threads=4,
                verbose=False,
            )
            self.model = "qwen2.5-3b-instruct"

    def _count_tokens(self, messages: list[dict]) -> int:
        """计算Chat格式messages的Token总数"""
        total_tokens = 0
        for msg in messages:
            # 每个message包含：role + content + 分隔符（参考OpenAI计数规则）
            total_tokens += len(self.tokenizer.encode(msg["content"])) + 4  # +4是role/分隔符开销
        return total_tokens

    def _truncate_messages(self, messages: list[dict]) -> list[dict]:
        """截断messages，确保输入Token ≤ max_ctx - reserved_output_tokens"""
        max_input_tokens = self.max_ctx - self.reserved_output_tokens
        current_tokens = self._count_tokens(messages)
        
        if current_tokens <= max_input_tokens:
            return messages
        
        # 仅截断最后一个user消息（通常是最长的prompt）
        user_msg = messages[-1]
        if user_msg["role"] != "user":
            logger.warning("非user消息超限，直接截断最后一条消息")
            truncated_msg = messages[:-1]
            return truncated_msg
        
        # 计算需要截断的Token数
        excess_tokens = current_tokens - max_input_tokens
        user_content = user_msg["content"]
        user_tokens = self.tokenizer.encode(user_content)
        
        # 截断user消息（保留开头，删除末尾）
        if len(user_tokens) <= excess_tokens:
            # 整条user消息都要删
            truncated_msg = messages[:-1]
        else:
            truncated_tokens = user_tokens[:-excess_tokens]
            truncated_content = self.tokenizer.decode(truncated_tokens)
            user_msg["content"] = truncated_content + "\n（内容已截断以适配模型上下文）"
            truncated_msg = messages
        
        logger.warning(
            f"Token超限：{current_tokens} > {max_input_tokens}，已截断至{self._count_tokens(truncated_msg)} Token"
        )
        return truncated_msg

    def generate(self, messages: list[dict]) -> str:
        # 第一步：截断超限的messages
        messages = self._truncate_messages(messages)
        
        if isinstance(self.llm, OpenAI):
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    response = self.llm.chat.completions.create(
                        messages=messages, 
                        temperature=0, 
                        model=self.model,
                        max_tokens=self.reserved_output_tokens  # 限制输出长度
                    )
                    return response.choices[0].message.content.strip()
                except Exception as e:
                    logger.error(f"OpenAI调用失败（尝试{attempt+1}/{max_retries}）：{e}")
                    if attempt == max_retries - 1:
                        return "Unknown"  # 兜底返回，避免程序崩溃
                    sleep(3)
        else:
            try:
                response = self.llm.create_chat_completion(
                    messages=messages,
                    temperature=0,
                    max_tokens=self.reserved_output_tokens  # 限制输出长度
                )
                return response["choices"][0]["message"]["content"].strip()
            except ValueError as e:
                # 兜底：Token仍超限/其他错误
                logger.error(f"Llama.cpp调用失败：{e}")
                return "Unknown"
            except Exception as e:
                logger.error(f"未知错误：{e}")
                return "Unknown"

def set_global_llm(api_key: str = None, base_url: str = None, model: str = None, lang: str = "English"):
    global GLOBAL_LLM
    GLOBAL_LLM = LLM(api_key=api_key, base_url=base_url, model=model, lang=lang)

def get_llm() -> LLM:
    if GLOBAL_LLM is None:
        logger.info("No global LLM found, creating a default one. Use `set_global_llm` to set a custom one.")
        set_global_llm()
    return GLOBAL_LLM
