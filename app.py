import os
import re
import json
import gradio as gr
from typing import List, Tuple, Dict, Iterator
import threading
from queue import Queue, Empty

# LangChain imports
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage


# Import our custom tools and templates
from tools import (
    list_video_files, get_video_info, extract_video_audio,
    extract_subtitle, sync_subtitles, cleanup_temp,
    list_temp_files, check_external_subtitle, copy_to_temp,
    cleanup_subtitle
)
# Create a mapping from tool names to the actual functions
tool_map = {
    "list_video_files": list_video_files,
    "get_video_info": get_video_info,
    "extract_video_audio": extract_video_audio,
    "extract_subtitle": extract_subtitle,
    "sync_subtitles": sync_subtitles,
    "cleanup_temp": cleanup_temp,
    "list_temp_files": list_temp_files,
    "check_external_subtitle": check_external_subtitle,
    "copy_to_temp": copy_to_temp,
    "cleanup_subtitle": cleanup_subtitle,
}

# --- LangChain Setup ---
api_key = os.getenv("OPENAI_API_KEY", "")
base_url = os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com/v1")
model = os.getenv("MODEL_NAME", "deepseek-chat")

llm = ChatOpenAI(model=model, temperature=0, api_key=api_key, base_url=base_url)
tools = [
    list_video_files, get_video_info, extract_video_audio,
    extract_subtitle, sync_subtitles, cleanup_temp,
    list_temp_files, check_external_subtitle, copy_to_temp,
    cleanup_subtitle
]

prompt = ChatPromptTemplate.from_messages([
    ("system", """你是一个用于批量处理视频字幕的AI项目经理。你的任务是将用户的自然语言指令转换成一个结构化的、可执行的JSON任务计划。

你的工作流程如下：
1. **理解用户意图**: 分析用户想要处理哪些文件（例如，使用 `list_video_files` 工具）以及具体的操作（例如，提取音视频、同步字幕等）
2. **收集信息**: 对于每个找到的视频，首先使用 `check_external_subtitle` 来检查是否存在外部字幕文件
   - 如果存在外部字幕，优先使用外部字幕，用 `copy_to_temp` 将其复制到临时目录，然后直接进行同步
   - 如果不存在外部字幕，才使用 `get_video_info` 和 `extract_subtitle` 来提取内置字幕
   - 只有当用户明确要求提取内置字幕时（如："提取字幕"、"用视频里的字幕"等），才跳过外部字幕检查
3. **生成JSON计划**: 你的最终输出必须是一个JSON对象，包含：
   - `tasks` 列表：每个任务包含 `source_file` 和 `steps` 列表
   - `global_steps` 列表：在所有视频处理完毕后执行的全局清理步骤（默认 `cleanup_subtitle` ，只有当用户明确要求才 `cleanup_temp`）

**JSON输出格式示例**:
```json
{{
  "tasks": [
    {{
      "source_file": "Episode/S01E01.mp4",
      "steps": [
        {{"tool": "extract_video_audio", "params": {{"input_path": "Episode/S01E01.mp4", "output_filename": "S01E01.mp4", "video_stream_index": 0, "audio_stream_index": 1}}}},
        {{"tool": "copy_to_temp", "params": {{"file_path": "Episode/S01E01.srt"}}}},
        {{"tool": "sync_subtitles", "params": {{"video_filename": "S01E01.mp4", "subtitle_filename": "S01E01.srt", "output_subtitle_name": "S01E01_synced.srt"}}}}
      ]
    }}
  ],
  "global_steps": [
    {{"tool": "cleanup_subtitle", "params": {{"temp_dir": "tmp"}}}}
  ]
}}
```

请严格遵守此JSON格式。不要执行实际的工具操作，只需规划出这些步骤即可。
"""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

agent = create_openai_tools_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

def agent_planner_thread(message: str, chat_history: List, queue: Queue):
    """
    Target for the agent planning thread.
    The agent's goal is to produce a JSON plan.
    """
    try:
        # We don't need the streaming callback here for the planner
        response = agent_executor.invoke(
            {"input": message, "chat_history": chat_history}
        )
        queue.put({"type": "plan", "content": response.get("output", "")})
    except Exception as e:
        queue.put({"type": "error", "content": f"抱歉，规划时遇到错误: {e}"})


def task_executor_thread(tasks: List[Dict], global_steps: List[Dict], queue: Queue):
    """
    Target for the task execution thread.
    Executes the plan provided by the agent.
    """
    total_tasks = len(tasks)
    for i, task in enumerate(tasks):
        source_file = task.get("source_file", "未知文件")
        queue.put({
            "type": "progress",
            "content": f"好的，我们开始处理第 {i+1} 个任务，目标文件是 `{source_file}`。\n"
        })

        for step_idx, step in enumerate(task.get("steps", [])):
            tool_name = step.get("tool")
            params = step.get("params", {})
            
            # 翻译工具名称为更自然的语言
            tool_translation = {
                "copy_to_temp": "复制文件到临时目录",
                "extract_video_audio": "提取视频和音频流",
                "extract_subtitle": "提取字幕文件",
                "sync_subtitles": "同步字幕时间轴",
                "cleanup_temp": "清理临时文件",
                "cleanup_subtitle": "清理字幕文件"
            }
            step_description = tool_translation.get(tool_name, f"执行 `{tool_name}`")

            queue.put({"type": "progress", "content": f"\n- **第 {step_idx + 1} 步**: {step_description}..."})
            
            if tool_name in tool_map:
                try:
                    tool_function = tool_map[tool_name]
                    result = tool_function.invoke(params)
                    
                    if isinstance(result, dict) and result.get("status") == "success":
                        queue.put({"type": "progress", "content": f" ✅ 成功！"})
                        if "message" in result:
                             queue.put({"type": "progress", "content": f" `{result['message']}`"})
                    elif isinstance(result, dict) and result.get("status") == "error":
                        error_message = result.get('message', '未知错误')
                        queue.put({"type": "progress", "content": f" ❌ 失败。错误信息: {error_message}"})
                        break
                    else:
                        result_str = json.dumps(result, ensure_ascii=False, indent=2) if isinstance(result, dict) else str(result)
                        queue.put({"type": "progress", "content": f" ✅ 操作完成，返回信息：\n```{result_str}```"})

                except Exception as e:
                    queue.put({"type": "progress", "content": f" ❌ 发生严重错误: {e}"})
                    break
            else:
                queue.put({"type": "progress", "content": f" ⚠️ **警告**: 未找到名为 `{tool_name}` 的工具。"})
        
        queue.put({"type": "progress", "content": f"\n\n---\n"})

    # 执行全局清理步骤（仅执行一次）
    if global_steps:
        queue.put({"type": "progress", "content": "\n### 🧹 全局清理阶段\n"})
        for step_idx, step in enumerate(global_steps):
            tool_name = step.get("tool")
            params = step.get("params", {})
            
            tool_translation = {
                "cleanup_temp": "清理临时文件",
                "cleanup_subtitle": "清理字幕文件"
            }
            step_description = tool_translation.get(tool_name, f"执行 `{tool_name}`")
            
            queue.put({"type": "progress", "content": f"\n- **全局步骤 {step_idx + 1}**: {step_description}..."})
            
            if tool_name in tool_map:
                try:
                    tool_function = tool_map[tool_name]
                    result = tool_function.invoke(params)
                    
                    if isinstance(result, dict) and result.get("status") == "success":
                        queue.put({"type": "progress", "content": f" ✅ 成功！"})
                        if "message" in result:
                             queue.put({"type": "progress", "content": f" `{result['message']}`"})
                    elif isinstance(result, dict) and result.get("status") == "error":
                        error_message = result.get('message', '未知错误')
                        queue.put({"type": "progress", "content": f" ❌ 失败。错误信息: {error_message}"})
                    else:
                        result_str = json.dumps(result, ensure_ascii=False, indent=2) if isinstance(result, dict) else str(result)
                        queue.put({"type": "progress", "content": f" ✅ 操作完成，返回信息：\n```{result_str}```"})
                except Exception as e:
                    queue.put({"type": "progress", "content": f" ❌ 发生严重错误: {e}"})
            else:
                queue.put({"type": "progress", "content": f" ⚠️ **警告**: 未找到名为 `{tool_name}` 的工具。"})

    queue.put({"type": "done", "content": "所有任务处理完毕！"})


def predict(message: str, history: List[Tuple[str, str]]) -> Iterator[Tuple[List[Tuple[str, str]], str, str]]:
    # 正确构建chat_history：从history中提取(user_message, ai_response)对
    chat_history = []
    for user_msg, ai_msg in history:
        if user_msg:  # 用户消息不为空
            chat_history.append(HumanMessage(content=user_msg))
        if ai_msg:  # AI响应不为空
            chat_history.append(AIMessage(content=ai_msg))
    
    q = Queue()
    log_content = "### 阶段一：任务规划\n"
    yield history[:-1] + [[message, "正在思考中，请稍候..."]], log_content

    # --- Planning Stage ---
    planner_thread = threading.Thread(target=agent_planner_thread, args=(message, chat_history, q))
    planner_thread.start()

    plan_json_str = ""
    while planner_thread.is_alive() or not q.empty():
        try:
            event = q.get(timeout=0.1)
            if event["type"] == "plan":
                plan_json_str = event["content"]
                break
            elif event["type"] == "error":
                yield history[:-1] + [[message, event["content"]]], log_content
                return
        except Empty:
            continue
    
    planner_thread.join()

    # 使用正则提取 JSON 内容
    json_match = re.search(r'\{.*\}', plan_json_str, re.DOTALL)
    
    if json_match:
        plan_json_str = json_match.group()
        log_content += f"✅ AI生成了任务计划。\n```json\n{plan_json_str}\n```\n"
        yield history[:-1] + [[message, "规划完成，准备执行..."]], log_content
        
        try:
            plan = json.loads(plan_json_str)
        except json.JSONDecodeError:
            log_content += f"❌ JSON解析失败，原始字符: {plan_json_str}\n"
            yield history[:-1] + [[message, "AI生成的计划格式有误，无法解析。"]], log_content
            return
    else:
        # 如果找不到 JSON 结构，说明 LLM 直接回复了对话
        log_content += f"⚠️ AI未生成结构化计划，{plan_json_str}\n"
        # 尝试直接把 LLM 的回复展示给用户
        yield history[:-1] + [[message, f"AI未生成执行计划，{plan_json_str}"]], log_content
        return
    
    yield history[:-1] + [[message, "规划完成，准备执行..."]], log_content

    # --- Execution Stage ---
    try:
        plan = json.loads(plan_json_str)
        tasks = plan.get("tasks", [])
        global_steps = plan.get("global_steps", [])
        
        # 检查是否有任何有效任务或全局步骤
        if not tasks and not global_steps:
            yield history[:-1] + [[message, "计划中没有发现任何有效任务或全局步骤。"]], log_content
            return

    except json.JSONDecodeError:
        yield history[:-1] + [[message, "无法解析AI生成的任务计划，请检查格式。"]], log_content
        return

    log_content += "\n### 阶段二：任务执行\n"
    
    executor_thread = threading.Thread(target=task_executor_thread, args=(tasks, global_steps, q))
    executor_thread.start()

    final_response = "任务执行中..."
    while executor_thread.is_alive() or not q.empty():
        try:
            event = q.get(timeout=0.1)
            if event["type"] == "progress":
                log_content += event["content"]
            elif event["type"] == "done":
                final_response = event["content"]
            
            yield history[:-1] + [[message, final_response]], log_content
        except Empty:
            continue
    
    executor_thread.join()
    yield history[:-1] + [[message, final_response]], log_content

# --- Gradio Interface ---
with gr.Blocks(
    title="✍️ 字幕时间线校正助手",
    theme="soft",
    css="""
    .markdown-container {
        padding: 10px 5px !important;
    }
    .markdown-label {
        font-weight: bold;
        margin-bottom: 10px;
        display: block;
    }
""",
) as app:

    gr.Markdown("# ✍️ 字幕时间线校正助手\n跟我说想要校正字幕的视频名称，我将为你搜索并进行提取音频流、字幕文件和时间线的校正、以及后续的清理工作...")

    with gr.Row():
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(label="对话窗口", height=500, show_copy_button=True)
            
            with gr.Row():
                user_input = gr.Textbox(
                    show_label=False, 
                    placeholder="指令如：'帮我同步狂飙第一集字幕时间轴...'", 
                    scale=8, 
                    container=False
                )
                submit_btn = gr.Button("发送", scale=1, variant="primary")
                clear_btn = gr.Button("清除", scale=1, variant="secondary")

        with gr.Column(scale=1):
            gr.HTML("<div class='markdown-label'>📝 处理步骤</div>")
            log_display = gr.Markdown(elem_classes=["markdown-container"], elem_id="log-box", height=400)
            
            # 自动滚动脚本：只在用户位于底部附近时自动滚动
            scroll_js = """
            function() {
                const el = document.querySelector('#log-box .markdown-container');
                if (el) {
                    // 检查用户是否在底部附近（距离底部5px以内）
                    const isNearBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 5;
                    
                    // 只有在接近底部时才自动滚动
                    if (isNearBottom) {
                        setTimeout(() => {
                            el.scrollTop = el.scrollHeight;
                        }, 50);
                    }
                }
            }
            """
            log_display.change(fn=None, inputs=None, outputs=None, js=scroll_js)

    def chat_fn(message: str, chat_history: List[Tuple[str, str]]) -> Iterator[Tuple[List[Tuple[str, str]], str, str]]:
        chat_history.append((message, ""))
        
        # 禁用输入框和按钮
        yield chat_history, "", gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=False)

        for updated_history, logs in predict(message, chat_history):
            chat_history = updated_history
            yield chat_history, logs, gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=False)
        
        # 处理完成后，启用输入框和按钮
        yield chat_history, logs, gr.update(interactive=True), gr.update(interactive=True), gr.update(interactive=True)

    submit_btn.click(
        chat_fn,
        inputs=[user_input, chatbot],
        outputs=[chatbot, log_display, user_input, submit_btn, clear_btn]
    ).then(lambda: gr.update(value=""), None, [user_input], queue=False)

    user_input.submit(
        chat_fn,
        inputs=[user_input, chatbot],
        outputs=[chatbot, log_display, user_input, submit_btn, clear_btn]
    ).then(lambda: gr.update(value=""), None, [user_input], queue=False)

    clear_btn.click(
        fn=lambda: ([], ""),
        inputs=[],
        outputs=[chatbot, log_display],
        queue=False
    )

if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=80)
