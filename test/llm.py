import json
import os
import time
from google import genai


# 创建文件目录
def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def save_llm_output(session_id: str, llm_output: dict):
    """
    将 LLM 输出保存到:
    video_processing_data/<session_id>/llm_output.json
    """
    base_dir = "video_processing_data"
    session_dir = os.path.join(base_dir, session_id)
    ensure_dir(session_dir)

    output_file = os.path.join(session_dir, "llm_output.json")

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(llm_output, f, indent=2, ensure_ascii=False)

    print(f"✅ LLM 结果已保存到: {output_file}")


def wait_for_file_active(client, file_resource, max_wait_time=120, wait_interval=5):
    """
    等待文件变为 ACTIVE 状态
    """
    print("等待文件处理完成...")
    waited_time = 0
    
    while waited_time < max_wait_time:
        try:
            file_status = client.files.get(name=file_resource.name)
            print(f"文件状态: {file_status.state} (等待 {waited_time} 秒)")
            
            if file_status.state == "ACTIVE":
                print("✅ 文件已就绪")
                return True
            elif file_status.state == "FAILED":
                raise Exception(f"文件处理失败: {file_status.state}")
            elif file_status.state == "PROCESSING":
                # 继续等待
                pass
            else:
                print(f"未知文件状态: {file_status.state}")
        
        except Exception as e:
            print(f"获取文件状态失败: {e}")
        
        time.sleep(wait_interval)
        waited_time += wait_interval
    
    raise Exception(f"文件处理超时（{max_wait_time}秒），请稍后重试")

def call_gemini(video_path: str, prompt: str, api_key: str,session_id: str, max_retries=3) -> dict[str, any]:
    """
    调用 Gemini 大模型，支持重试机制
    """
    for attempt in range(max_retries):
        try:
            print(f"第 {attempt + 1} 次尝试调用Gemini...")
            
            # 初始化 Gemini 客户端
            client = genai.Client(api_key=api_key)

            # 检查文件是否存在
            if not os.path.exists(video_path):
                raise FileNotFoundError(f"视频文件不存在: {video_path}")
            
            # 检查文件大小
            file_size = os.path.getsize(video_path) / (1024 * 1024)
            print(f"视频文件大小: {file_size:.2f} MB")

            # 上传视频文件到Gemini
            print("正在上传视频文件...")
            myfile = client.files.upload(file=video_path)
            print(f"文件上传成功，文件ID: {myfile.uri}")

            # 等待文件变为 ACTIVE 状态
            wait_for_file_active(client, myfile)

            # 构建系统提示词
            system_prompt = f"""
你是一个视频处理策略生成器。请根据视频内容和以下指令输出严格格式化的JSON：
{prompt}

你是一个视频预处理智能调度代理（Video Preprocessing Agent）。

你的任务是：
1️⃣ 根据视频情况，决定需要执行哪些预处理模块；
2️⃣ 为接收端生成一份视频生成模型的选择策略（generate_policy）；
3️⃣ 生成用于视频生成的自然语言提示词（prompt_generation）。

请综合分析这些特征，自动判断视频类别，并输出最优的处理与压缩策略。 常见类别与处理策略如下： 
1. 竖屏短视频 → 使用 keyframe_extraction 压缩策略 
2. 高清长视频 → 使用 h264_crf 压缩，crf = 32 
3. 语音/讲座类视频 → 使用 audio2text + metadata 
4. 高帧率运动视频 → 使用 motion_token 分析 
5. 监控类视频 → 使用 motion_token + metadata 
6. 低分辨率视频 → 轻量化压缩，仅提取关键帧和语音

系统中可用的工具如下：
- video2metadata：提取视频元数据
- video2pictures：视频转图像
- picturepicker：关键帧选择
- video2audio：提取音频
- audio2text：语音转录（仅当有语音内容时启用）
- video2motionTokens：运动特征提取

输出要求：
- 严格按照以下 JSON 结构输出；
- 发送端只会执行 process_policy 中为 true 的工具；
- 其余字段仅供接收端使用。

-----------------------------------
输出结构如下：
{{
  "llm_output": {{
    "process_policy": {{
      "video2pictures": true/false,
      "video2audio": true/false,
      "picturepicker": true/false,
      "audio2text": true/false,
      "video2metadata": true/false,
      "video2motionTokens": true/false
    }},
    "generate_policy": {{
      "model_name": "VideoCrafter2",
      "module": "videocrafter2_generator",
      "expected_inputs": ["prompt", "keyframes", "motion_tokens", "metadata"],
      "reasoning": "解释为什么选择该模型。"
    }},
    "prompt_generation": {{
      "prompt": "对视频内容的自然语言描述。"
    }}
  }}
}}
不要包含任何解释性文字，只输出JSON。
"""

            # 调用 Gemini 模型
            print("正在调用Gemini模型分析视频...")
            response = client.models.generate_content(
                model="gemini-2.5-flash", 
                contents=[myfile, system_prompt]
            )

            # 提取模型输出
            result_text = response.text.strip()
            print(f"Gemini原始响应: {result_text}")

            # 清理和解析JSON
            cleaned_text = result_text
            if "```json" in cleaned_text:
                cleaned_text = cleaned_text.split("```json")[1].split("```")[0].strip()
            
            cleaned_text = cleaned_text.replace("{{", "{").replace("}}", "}")
            
            if "{" in cleaned_text and "}" in cleaned_text:
                start = cleaned_text.find('{')
                end = cleaned_text.rfind('}') + 1
                cleaned_text = cleaned_text[start:end]
            
            policy = json.loads(cleaned_text)

            # ⭐⭐ 保存 llm_output.json ⭐⭐
            llm_output = policy.get("llm_output", {})
            save_llm_output(session_id, llm_output)

            return policy

        except json.JSONDecodeError as e:
            print(f"JSON解析错误: {e}")
            print(f"原始返回: {result_text}")
            if attempt == max_retries - 1:
                return get_fallback_policy()
            print("重试中...")
            time.sleep(5)
            
        except Exception as e:
            print(f"Gemini API 调用错误: {e}")
            if attempt == max_retries - 1:
                return get_fallback_policy()
            print("重试中...")
            time.sleep(5)
    
    return get_fallback_policy()
    
def get_fallback_policy() -> dict[str, any]:
    """
    返回默认策略（当模型或网络调用失败时）
    """
    return {
        "llm_output": {
            "process_policy": {
                "video2pictures": True,
                "video2audio": True,
                "picturepicker": False,
                "audio2text": True,
                "video2metadata": False,
                "video2motionTokens": False
            },
            "generate_policy": {
                "model_name": "VideoCrafter2",
                "module": "videocrafter2_generator",
                "expected_inputs": ["prompt", "keyframes"],
                "reasoning": "默认策略：提取关键帧和音频进行基本视频生成"
            },
            "prompt_generation": {
                "prompt": "视频内容分析失败，使用默认处理策略"
            }
        }
    }