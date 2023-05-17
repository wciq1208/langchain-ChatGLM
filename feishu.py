# coding=utf8
import json
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Dict

import nltk
import requests

from chains.local_doc_qa import LocalDocQA
from configs.model_config import *

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
    def send_api(url, data, headers: dict | None = None, timeout=None):
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
        return resp, code, err


class FeishuServer(BaseHTTPRequestHandler):
    feishu_client = None
    local_doc_qa = None
    vs_path = None

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

    @classmethod
    def get_answer(cls, query, history):
        print(11111)
        if cls.vs_path is not None and os.path.exists(cls.vs_path):
            print(22222)
            for resp, history in cls.local_doc_qa.get_knowledge_based_answer(
                    query=query, vs_path=cls.vs_path, chat_history=history, streaming=False):
                print("####")
                print(resp)
                print("=======")
                print(history)
                #yield history, ""
        logger.info(
            f"flagging: username={FLAG_USER_NAME},query={query},vs_path={cls.vs_path},history={history}")

    @classmethod
    def event(cls, data):
        event = data.get("event")
        msg = event.get("message", dict())
        msg_id = msg.get("message_id")
        content: str = json.loads(msg.get("content", "{}")).get("text", "")
        at_list = []
        for user in msg.get("mentions", []):
            at_list.append(user.get("key", None))
        for user in at_list:
            if user is not None:
                content = content.replace(user, "")
        content = content.strip()
        cls.get_answer(content, [])
        return cls.feishu_client.reply(msg_id, "test")

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


def run():
    logger.info("start init")
    FeishuServer.init("test")
    server = HTTPServer(("0.0.0.0", 7777), FeishuServer)
    logger.info("start server")
    server.serve_forever()


if __name__ == "__main__":
    run()
