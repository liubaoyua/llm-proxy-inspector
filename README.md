# LLM Proxy Inspector

OpenAI-compatible 反向代理 + 请求/响应可视化查看器。

## 安装

```bash
pip install -r requirements.txt
```

## 启动

```bash
# 默认：上游 http://127.0.0.1:8000，代理 :7654，UI :7655
python proxy.py

# 自定义
python proxy.py --upstream http://127.0.0.1:8000 --proxy-port 7654 --ui-port 7655
```

## 使用

- 客户端将 API 地址指向 `http://<your-host>:7654`
- 浏览器打开 `http://<your-host>:7655` 查看请求/响应

## 截图

**消息双栏视图（Request / Response）**

![消息视图](docs/message.png)

**Raw JSON 视图**

![Raw JSON](docs/rawjson.png)

**SSE Chunks 视图**

![SSE Chunks](docs/sse-chunks.png)

## 功能

- [x] 透传所有 HTTP 方法，原始数据不变
- [x] 流式 SSE 实时转发，结束后自动合并解析
- [x] 非流式 JSON 响应直接展示
- [x] 消息双栏视图（Request / Response）
- [x] 思考链（reasoning）折叠展示
- [x] Raw JSON 视图，支持一键复制
- [x] 侧边栏5秒局部刷新，不影响当前 tab
- [x] URL 格式 `/ids/<record_id>` 可分享

## License

[MIT](LICENSE)

## 目录结构

```
llm-proxy/
├── proxy.py          # 主程序（代理 + UI 服务）
├── requirements.txt
└── static/
    └── index.html    # 单文件前端
```
