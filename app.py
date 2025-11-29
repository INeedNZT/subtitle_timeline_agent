import os
import re
import json
import gradio as gr
from typing import List, Tuple
import threading
import uuid
import time

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
1. **理解用户意图**: 分析用户想要处理哪些文件（例如，使用 `list_video_files` 工具）
2. **收集信息**: 对于每个找到的视频，除非用户明确要求，否则使用 `check_external_subtitle` 来检查是否存在外部字幕文件
   - 如果存在外部字幕，优先使用外部字幕，用 `copy_to_temp` 将其复制到临时目录
   - 如果不存在外部字幕，才使用 `get_video_info` 和 `extract_subtitle` 来提取内置字幕
3. **提取音视频**: 使用 `get_video_info` 分析视频元数据，通过 `extract_video_audio` 来根据流索引提取音视频
4. **同步字幕时间轴**: 通过 `sync_subtitles` 将视频文件和字幕文件进行时间轴的校正
5. **生成JSON计划**: 你的最终输出必须是一个JSON对象，包含：
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

# --- Global Job Management ---
JOBS = {}
JOBS_LOCK = threading.Lock()

def log_to_job(job_id: str, content: str, message_type: str = "progress"):
    """Helper to safely append logs to a job."""
    with JOBS_LOCK:
        if job_id in JOBS:
            JOBS[job_id]["logs"].append(content)
            if message_type in ["done", "error"]:
                JOBS[job_id]["status"] = message_type
                # 如果是完成或错误，也将此作为最终响应记录，以便更新UI
                JOBS[job_id]["final_response"] = content if message_type == "done" else f"出错: {content}"

def background_task_runner(job_id: str, message: str, chat_history_tuples: List[Tuple[str, str]]):
    """
    The main function running in a background thread.
    It handles both planning and execution phases.
    """
    # Construct LangChain history objects
    chat_history = []
    for user_msg, ai_msg in chat_history_tuples:
        if user_msg:
            chat_history.append(HumanMessage(content=user_msg))
        if ai_msg:
            chat_history.append(AIMessage(content=ai_msg))

    log_to_job(job_id, "### 阶段一：任务规划\n正在思考中，请稍候...\n")

    # --- Phase 1: Planning ---
    try:
        response = agent_executor.invoke(
            {"input": message, "chat_history": chat_history}
        )
        plan_json_str = response.get("output", "")
    except Exception as e:
        log_to_job(job_id, f"规划阶段发生异常: {e}", "error")
        return

    # Parse JSON Plan
    json_match = re.search(r'\{.*\}', plan_json_str, re.DOTALL)
    if json_match:
        plan_json_str = json_match.group()
        log_to_job(job_id, f"✅ AI生成了任务计划。\n```json\n{plan_json_str}\n```\n")
        
        try:
            plan = json.loads(plan_json_str)
        except json.JSONDecodeError:
            log_to_job(job_id, f"❌ JSON解析失败，原始字符: {plan_json_str}\n", "error")
            return
    else:
        log_to_job(job_id, f"⚠️ AI未生成结构化计划，{plan_json_str}\n", "done") # Treat as done if no plan
        return

    # --- Phase 2: Execution ---
    tasks = plan.get("tasks", [])
    global_steps = plan.get("global_steps", [])

    if not tasks and not global_steps:
        log_to_job(job_id, "计划中没有发现任何有效任务或全局步骤。", "done")
        return

    log_to_job(job_id, "\n### 阶段二：任务执行\n")

    for i, task in enumerate(tasks):
        source_file = task.get("source_file", "未知文件")
        log_to_job(job_id, f"好的，我们开始处理第 {i+1} 个任务，目标文件是 `{source_file}`。\n")

        for step_idx, step in enumerate(task.get("steps", [])):
            tool_name = step.get("tool")
            params = step.get("params", {})
            
            tool_translation = {
                "copy_to_temp": "复制文件到临时目录",
                "extract_video_audio": "提取视频和音频流",
                "extract_subtitle": "提取字幕文件",
                "sync_subtitles": "同步字幕时间轴",
            }
            step_description = tool_translation.get(tool_name, f"执行 `{tool_name}`")
            log_to_job(job_id, f"\n- **第 {step_idx + 1} 步**: {step_description}...")

            if tool_name in tool_map:
                try:
                    tool_function = tool_map[tool_name]
                    result = tool_function.invoke(params)
                    
                    if isinstance(result, dict) and result.get("status") == "success":
                        log_to_job(job_id, " ✅ 成功！")
                        if "message" in result:
                             log_to_job(job_id, f" `{result['message']}`")
                    elif isinstance(result, dict) and result.get("status") == "error":
                        error_message = result.get('message', '未知错误')
                        log_to_job(job_id, f" ❌ 失败。错误信息: {error_message}")
                        # Optionally break the loop on error? For now, we continue or break task
                        break 
                    else:
                        result_str = json.dumps(result, ensure_ascii=False, indent=2) if isinstance(result, dict) else str(result)
                        log_to_job(job_id, f" ✅ 操作完成，返回信息：\n```{result_str}```")
                except Exception as e:
                    log_to_job(job_id, f" ❌ 发生严重错误: {e}")
                    break
            else:
                log_to_job(job_id, f" ⚠️ **警告**: 未找到名为 `{tool_name}` 的工具。")
        
        log_to_job(job_id, "\n\n---\n")

    # Global Steps
    if global_steps:
        log_to_job(job_id, "\n### 🧹 全局清理阶段\n")
        for step_idx, step in enumerate(global_steps):
            tool_name = step.get("tool")
            params = step.get("params", {})
            
            tool_translation = {
                "cleanup_temp": "清理临时目录",
                "cleanup_subtitle": "清理字幕文件"
            }
            step_description = tool_translation.get(tool_name, f"执行 `{tool_name}`")
            log_to_job(job_id, f"\n- **全局步骤 {step_idx + 1}**: {step_description}...")
            
            if tool_name in tool_map:
                try:
                    tool_function = tool_map[tool_name]
                    result = tool_function.invoke(params)
                    if isinstance(result, dict) and result.get("status") == "success":
                        log_to_job(job_id, " ✅ 成功！")
                    elif isinstance(result, dict) and result.get("status") == "error":
                        log_to_job(job_id, f" ❌ 失败: {result.get('message')}")
                    else:
                         log_to_job(job_id, f" ✅ 完成")
                except Exception as e:
                    log_to_job(job_id, f" ❌ 错误: {e}")
            else:
                log_to_job(job_id, f" ⚠️ 未知工具: {tool_name}")

    log_to_job(job_id, "所有任务处理完毕！", "done")


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

    # State to store the current job ID
    job_id_state = gr.State("")

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
            
            # 自动滚动脚本
            scroll_js = """
            function() {
                const el = document.querySelector('#log-box .markdown-container');
                if (el) {
                    const isNearBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 10;
                    if (isNearBottom) {
                        setTimeout(() => { el.scrollTop = el.scrollHeight; }, 50);
                    }
                }
            }
            """
            log_display.change(fn=None, inputs=None, outputs=None, js=scroll_js)

    def start_task(message: str, history: List[Tuple[str, str]]):
        """
        Starts the background task and returns the initial UI state.
        """
        if not message.strip():
            return history, "", gr.update(), gr.update(), gr.update(), ""

        # Create a new Job
        new_job_id = str(uuid.uuid4())
        with JOBS_LOCK:
            JOBS[new_job_id] = {
                "status": "running",
                "logs": [],
                "final_response": None,
                "created_at": time.time()
            }

        # Update chat history with user message
        history.append((message, "🚀 任务已提交至后台，正在处理中..."))

        # Start the background thread
        threading.Thread(
            target=background_task_runner, 
            args=(new_job_id, message, history[:-1]), # Pass history excluding current new message
            daemon=True
        ).start()

        # Return updated UI state
        # Disable inputs while processing
        return (
            history, 
            "", 
            gr.update(interactive=False), 
            gr.update(interactive=False), 
            gr.update(interactive=False),
            new_job_id # Set the state
        )

    def monitor_task(job_id: str, history: List[Tuple[str, str]]):
        """
        Generator that yields log updates for the given job_id.
        """
        if not job_id:
            yield history, "", gr.update(interactive=True), gr.update(interactive=True), gr.update(interactive=True)
            return

        last_log_count = 0
        
        while True:
            job_data = None
            with JOBS_LOCK:
                job_data = JOBS.get(job_id)
            
            if not job_data:
                # Job not found
                yield history, "⚠️ 找不到任务信息。", gr.update(interactive=True), gr.update(interactive=True), gr.update(interactive=True)
                break
            
            # Get current logs
            current_logs = job_data["logs"]
            full_log_text = "".join(current_logs)
            
            # Check status
            status = job_data.get("status", "running")
            
            if status in ["done", "error"]:
                # Update final chat message
                final_resp = job_data.get("final_response", "任务结束")
                # Update the last AI message in history
                if history:
                    history[-1] = (history[-1][0], final_resp)
                
                yield history, full_log_text, gr.update(interactive=True), gr.update(interactive=True), gr.update(interactive=True)
                break
            else:
                # Still running, just update logs
                # Only yield if logs have changed to save bandwidth/rendering? 
                # Gradio handles frequent yields okay, but checking count is better.
                if len(current_logs) > last_log_count:
                    yield history, full_log_text, gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=False)
                    last_log_count = len(current_logs)
            
            time.sleep(0.5) # Poll interval

    # Wire up events
    # 1. Submit -> Start Task (updates history, disables inputs, sets job_id)
    # 2. Then -> Monitor Task (reads job_id, updates logs and history, re-enables inputs when done)
    
    submit_event = submit_btn.click(
        start_task,
        inputs=[user_input, chatbot],
        outputs=[chatbot, user_input, submit_btn, clear_btn, user_input, job_id_state] # Note: user_input listed twice to clear it and disable it? Actually outputs map positionally.
        # wait, start_task outputs: history, user_input_val, submit_interactive, clear_interactive, user_interactive, job_id
    )
    
    submit_event.then(
        monitor_task,
        inputs=[job_id_state, chatbot],
        outputs=[chatbot, log_display, submit_btn, clear_btn, user_input]
    )

    # Handle 'Enter' key in textbox
    enter_event = user_input.submit(
        start_task,
        inputs=[user_input, chatbot],
        outputs=[chatbot, user_input, submit_btn, clear_btn, user_input, job_id_state]
    )
    
    enter_event.then(
        monitor_task,
        inputs=[job_id_state, chatbot],
        outputs=[chatbot, log_display, submit_btn, clear_btn, user_input]
    )

    clear_btn.click(
        fn=lambda: ([], ""),
        inputs=[],
        outputs=[chatbot, log_display]
    )

if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=80)
