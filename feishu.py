# coding=utf8
import json
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Dict

import nltk
import requests

from langchain.vectorstores import FAISS

from chains.local_doc_qa import LocalDocQA
from chains.local_doc_qa import similarity_search_with_score_by_vector, generate_prompt
from configs.model_config import *
from utils import torch_gc

nltk.data.path = [NLTK_DATA_PATH] + nltk.data.path
stream_log = logging.StreamHandler()
stream_log.setLevel(logging.INFO)
logger.addHandler(stream_log)
logger.setLevel(logging.INFO)


class FeishuClient:
    APP_ID = "cli_a4ee807bc27e500e"
    APP_SECRET = "JN6Y9mhyUgiiHcvW4I7wQdTCV6q5ZWAM"

    def __init__(self):
        self.feishu_token = None
        self.reset_feishu_token()

    @staticmethod
    def send_api(url, data, headers=None, timeout=None):
        try:
            resp = requests.post(url, json=data, timeout=timeout, headers=headers)
            msg: Dict = resp.json()
            if msg["code"] != 0:
                return msg, msg["code"], msg["msg"]
            return msg, msg["code"], None
        except Exception as e:
            return None, -1, str(e)

    def reset_feishu_token(self):
        url = "https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal/"
        token, code, err = FeishuClient.send_api(url, {
            "app_id": FeishuClient.APP_ID,
            "app_secret": FeishuClient.APP_SECRET
        })
        if code == 0:
            token = token.get("tenant_access_token")
            if token is not None:
                self.feishu_token = token
            else:
                logger.info("token is empty")
        else:
            logger.info(err)

    def reply(self, msg_id, content):
        url = f"https://open.feishu.cn/open-apis/im/v1/messages/{msg_id}/reply"
        content = json.dumps({
            "text": content
        })
        while True:
            resp, code, err = FeishuClient.send_api(url, {
                "content": content,
                "msg_type": "text",
            }, {
                                                        "Authorization": f"Bearer {self.feishu_token}"
                                                    })
            if code != 99991661:
                break
            self.reset_feishu_token()
        if code != 0:
            logger.error(code, err)
        return resp, code, err


class FeishuServer(BaseHTTPRequestHandler):
    feishu_client = None
    local_doc_qa = None
    vs_path = None
    reply_msg_id_map = dict()
    vector_store = None

    @classmethod
    def get_knowledge_based_answer(cls, query, chat_history=[], streaming: bool = STREAMING):
        related_docs_with_score = cls.vector_store.similarity_search_with_score(query, k=cls.local_doc_qa.top_k)
        torch_gc()
        prompt = generate_prompt(related_docs_with_score, query)

        for result, history in cls.local_doc_qa.llm._call(prompt=prompt,
                                                          history=chat_history,
                                                          streaming=streaming):
            torch_gc()
            history[-1][0] = query
            response = {"query": query,
                        "result": result,
                        "source_documents": related_docs_with_score}
            yield response, history
            torch_gc()

    @classmethod
    def _init_vector_store(cls):
        cls.vector_store = FAISS.load_local(cls.vs_path, cls.local_doc_qa.embeddings)
        FAISS.similarity_search_with_score_by_vector = similarity_search_with_score_by_vector
        cls.vector_store.chunk_size = cls.local_doc_qa.chunk_size
        cls.vector_store.chunk_conent = cls.local_doc_qa.chunk_conent
        cls.vector_store.score_threshold = cls.local_doc_qa.score_threshold

    @classmethod
    def _init_model(cls):
        try:
            cls.local_doc_qa.init_cfg()
            cls.local_doc_qa.llm._call("你好")
            reply = """模型已成功加载，可以开始对话，或从右侧选择模式后开始对话"""
            logger.info(reply)
        except Exception as e:
            logger.error(e)
            reply = """模型未成功加载，请到页面左上角"模型配置"选项卡中重新选择后点击"加载模型"按钮"""
            if str(e) == "Unknown platform: darwin":
                logger.info("该报错可能因为您使用的是 macOS 操作系统，需先下载模型至本地后执行，具体方法请参考项目 README 中本地部署方法及常见问题："
                            " https://github.com/imClumsyPanda/langchain-ChatGLM")
            else:
                logger.info(reply)

    @classmethod
    def init(cls, doc_name):
        cls.vs_path = os.path.join(VS_ROOT_PATH, doc_name)
        cls.feishu_client = FeishuClient()
        cls.local_doc_qa = LocalDocQA()
        cls._init_model()
        cls._init_vector_store()

    @classmethod
    def get_answer(cls, query, history):
        result = ""
        if cls.vs_path is not None and os.path.exists(cls.vs_path):
            for resp, history in cls.get_knowledge_based_answer(
                    query=query, chat_history=history, streaming=False):
                result += resp.get("result", "") + "\n"
        logger.info(
            f"flagging: username={FLAG_USER_NAME},query={query},vs_path={cls.vs_path},history={history}")
        return result.strip()

    @classmethod
    def event(cls, data):
        event = data.get("event")
        msg = event.get("message", dict())
        msg_id = msg.get("message_id")
        logger.info(f"get msg id:{msg_id}")
        if msg_id in cls.reply_msg_id_map:
            return None, 0, None
        cls.reply_msg_id_map[msg_id] = time.time()
        content: str = json.loads(msg.get("content", "{}")).get("text", "")
        at_list = []
        for user in msg.get("mentions", []):
            at_list.append(user.get("key", None))
        for user in at_list:
            if user is not None:
                content = content.replace(user, "")
        content = content.strip()
        result = cls.get_answer(content, [])
        return cls.feishu_client.reply(msg_id, result)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        data = json.loads(body)
        self.send_response(200)
        self.send_header("Content-type", "application/json")
        self.end_headers()
        if data.get("type") == "url_verification":
            self.wfile.write(json.dumps({"challenge": data.get("challenge")}).encode("utf-8"))
        else:
            resp, code, err = FeishuServer.event(data)
            if err is not None:
                logger.info(err)

    @classmethod
    def clean_msg_id_map(cls):
        while True:
            now = time.time()
            clean_list = set()
            for msg_id, t in cls.reply_msg_id_map.items():
                if now > 600 + t:
                    clean_list.add(msg_id)
            for msg_id in clean_list:
                del cls.reply_msg_id_map[msg_id]
            time.sleep(300)


def run():
    logger.info("start init")
    FeishuServer.init("test")
    th = threading.Thread(target=FeishuServer.clean_msg_id_map)
    th.start()
    server = HTTPServer(("0.0.0.0", 7777), FeishuServer)
    logger.info("start server")
    server.serve_forever()


if __name__ == "__main__":
    run()
