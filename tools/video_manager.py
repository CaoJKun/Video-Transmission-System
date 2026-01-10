import os
import shutil
import json
import threading
import time
import logging
import base64
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict

logger = logging.getLogger(__name__)

# 支持的文件格式
SUPPORTED_VIDEO_FORMATS = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.webm', '.m4v'}
SUPPORTED_AUDIO_FORMATS = {'.wav', '.mp3', '.aac', '.flac', '.m4a'}

class SessionCleanupManager:
    """会话清理管理器 - 自动清理过期会话"""
    
    def __init__(self, tools_manager, cleanup_interval_hours: int = 1, session_ttl_hours: int = 24):
        self.tools_manager = tools_manager
        self.cleanup_interval = timedelta(hours=cleanup_interval_hours)
        self.session_ttl = timedelta(hours=session_ttl_hours)
        self.running = True
        self.cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self.cleanup_thread.start()
        logger.info(f"会话清理管理器启动: 清理间隔 {cleanup_interval_hours}h, 会话TTL {session_ttl_hours}h")
    
    def _cleanup_loop(self):
        """清理循环"""
        while self.running:
            try:
                self._cleanup_expired_sessions()
                time.sleep(self.cleanup_interval.total_seconds())
            except Exception as e:
                logger.error(f"会话清理出错: {e}")
    
    def _cleanup_expired_sessions(self):
        """清理过期会话"""
        now = datetime.now()
        expired_sessions = []
        
        with self.tools_manager.lock:
            for session_id, session_data in self.tools_manager.sessions.items():
                session_time = session_data.get("created_at", now)
                if now - session_time > self.session_ttl:
                    expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            try:
                self.tools_manager.cleanup_session(session_id)
                logger.info(f"自动清理过期会话: {session_id}")
            except Exception as e:
                logger.error(f"清理会话失败 {session_id}: {e}")
    
    def stop(self):
        """停止清理任务"""
        self.running = False
        logger.info("会话清理管理器已停止")

class VideoToolsManager:
    """视频工具管理器 - 处理文件存储和工具间的数据传递"""
    
    def __init__(self, work_dir_path: str = None):
        self.sessions = {}  # 存储会话数据 {session_id: {file_data, processed_data, created_at}}
        self.lock = threading.RLock()  # 线程安全锁
        self.work_dir = Path(work_dir_path)
    
    def create_session(self, file_data: str, filename: str) -> str:
        """创建新的处理会话"""
        with self.lock:
            session_id = str(uuid.uuid4())
            extension = Path(filename).suffix.lower()
            
            # 验证文件格式
            if extension not in SUPPORTED_VIDEO_FORMATS:
                raise ValueError(f"不支持的视频格式: {extension}")
            
            # 验证 base64 数据
            try:
                decoded_size = len(base64.b64decode(file_data))
            except Exception:
                raise ValueError("无效的 base64 数据")
            
            # 为每个处理会话创建独立的工作目录
            session_work_dir = self.work_dir / session_id
            session_work_dir.mkdir(parents=True, exist_ok=True)
            
            # 保存原始视频文件到会话目录
            original_video_path = session_work_dir / f"original{extension}"
            with open(original_video_path, "wb") as f:
                f.write(base64.b64decode(file_data))

            # 创建会话子目录结构
            subdirs = ["frames", "keyframes", "audio", "temp", "exports"]
            for subdir in subdirs:
                (session_work_dir / subdir).mkdir(exist_ok=True)
            
            self.sessions[session_id] = {
                "original_file": {
                    "data": file_data,
                    "filename": filename,
                    "extension": extension,
                    "size": decoded_size,
                    "path": str(original_video_path)
                },
                "processed_data": {},
                "created_at": datetime.now(),
                "work_dir": str(session_work_dir)
            }
            logger.info(f"创建新会话: {session_id}, 文件: {filename}, 工作目录: {session_work_dir}")
            return session_id
    
    def get_session_data(self, session_id: str, data_key: str = None):
        """获取会话数据"""
        with self.lock:
            if session_id not in self.sessions:
                # 尝试从文件系统加载会话数据
                self._load_session_from_filesystem(session_id)
                
            if session_id not in self.sessions:
                logger.warning(f"会话不存在: {session_id}")
                return None
            session = self.sessions[session_id]
            if data_key:
                value = session["processed_data"].get(data_key)
                # 若指定键不存在，尝试从磁盘回填（支持 video_tokens 与 motion_tokens）
                if value is None:
                    work_dir = Path(session.get("work_dir", ""))
                    try:
                        if data_key == "video_tokens":
                            tokens_path = work_dir / "video_tokens.json"
                            if tokens_path.exists():
                                with open(tokens_path, "r", encoding="utf-8") as f:
                                    tokens_data = json.load(f)
                                if isinstance(tokens_data, dict) and all(k in tokens_data for k in ["tokens_shape", "tokens_data", "tokens_dtype"]):
                                    session["processed_data"]["video_tokens"] = tokens_data
                                    logger.info(f"已按需从磁盘加载 video_tokens: {tokens_path}")
                                    return tokens_data
                        elif data_key == "motion_tokens":
                            tokens_path = work_dir / "motion_tokens.json"
                            if tokens_path.exists():
                                with open(tokens_path, "r", encoding="utf-8") as f:
                                    motion_tokens = json.load(f)
                                session["processed_data"]["motion_tokens"] = motion_tokens
                                logger.info(f"已按需从磁盘加载 motion_tokens: {tokens_path}")
                                return motion_tokens
                    except Exception as e:
                        logger.warning(f"按需加载 {data_key} 失败: {e}")
                return value
            return session
    
    def _load_session_from_filesystem(self, session_id: str):
        """从文件系统加载会话数据"""
        session_dir = self.work_dir / session_id
        if not session_dir.exists():
            return False
        
        # 检查是否有原始视频文件
        original_file = session_dir / "original.mp4"
        if not original_file.exists():
            return False
        
        # 重建会话数据结构
        self.sessions[session_id] = {
            "original_file": {
                "path": str(original_file),
                "filename": "original.mp4",
                "extension": ".mp4",
                "size": original_file.stat().st_size
            },
            "processed_data": {},
            "created_at": None,  # 从session_info.json加载
            "work_dir": str(session_dir)
        }
        
        logger.info(f"从文件系统加载会话: {session_id}")
        # 额外尝试加载已持久化的处理数据（如 video_tokens）
        try:
            tokens_path = session_dir / "video_tokens.json"
            if tokens_path.exists():
                with open(tokens_path, "r", encoding="utf-8") as f:
                    tokens_data = json.load(f)
                # 简单校验必要字段
                if isinstance(tokens_data, dict) and all(k in tokens_data for k in ["tokens_shape", "tokens_data", "tokens_dtype"]):
                    self.sessions[session_id]["processed_data"]["video_tokens"] = tokens_data
                    logger.info(f"已从磁盘加载 video_tokens: {tokens_path}")
        except Exception as e:
            logger.warning(f"加载 video_tokens 失败: {e}")
        return True
    
    def get_session_work_dir(self, session_id: str) -> Path:
        """获取会话工作目录"""
        session = self.get_session_data(session_id)
        if session:
            return Path(session["work_dir"])
        return None
    
    def save_processed_data(self, session_id: str, data_key: str, data):
        """保存处理后的数据到会话目录，同时返回给客户端"""
        with self.lock:
            if session_id not in self.sessions:
                logger.warning(f"会话不存在: {session_id}")
                return False
            
            session = self.sessions[session_id]
            work_dir = Path(session["work_dir"])
            
            # 根据数据类型保存到不同子目录，并确保数据也返回给客户端
            try:
                if data_key == "frames":
                    frames_dir = work_dir / "frames"
                    # 如果帧文件来自临时目录，移动到会话目录
                    if "frame_files" in data and "output_dir" in data:
                        new_frame_files = []
                        for i, old_path in enumerate(data["frame_files"]):
                            if os.path.exists(old_path):
                                new_path = frames_dir / f"frame_{i:04d}.jpg"
                                shutil.move(old_path, new_path)
                                new_frame_files.append(str(new_path))
                        data["frame_files"] = new_frame_files
                        # 清理原临时目录
                        temp_output_dir = data["output_dir"]
                        if os.path.exists(temp_output_dir) and temp_output_dir != str(frames_dir):
                            shutil.rmtree(temp_output_dir)
                        data["output_dir"] = str(frames_dir)
                
                elif data_key == "keyframes":
                    keyframes_dir = work_dir / "keyframes"
                    # 保存关键帧到独立文件
                    if "selected_frames" in data:
                        saved_keyframes = []
                        for i, frame_info in enumerate(data["selected_frames"]):
                            if "data" in frame_info:
                                keyframe_path = keyframes_dir / f"keyframe_{i:04d}.jpg"
                                with open(keyframe_path, "wb") as f:
                                    f.write(base64.b64decode(frame_info["data"]))
                                
                                # 为客户端保留原始数据
                                frame_copy = frame_info.copy()
                                frame_copy["saved_path"] = str(keyframe_path)
                                saved_keyframes.append(frame_copy)
                        data["saved_keyframes"] = saved_keyframes
                
                elif data_key == "audio":
                    audio_dir = work_dir / "audio"
                    # 移动音频文件到会话目录
                    if "audio_path" in data:
                        old_path = data["audio_path"]
                        if os.path.exists(old_path):
                            new_path = audio_dir / Path(old_path).name
                            shutil.move(old_path, new_path)
                            data["audio_path"] = str(new_path)
                
                elif data_key == "transcript":
                    # 保存转录文本到文件
                    transcript_path = work_dir / "transcript.txt"
                    if "text" in data:
                        with open(transcript_path, "w", encoding="utf-8") as f:
                            f.write(data["text"])
                        data["transcript_file"] = str(transcript_path)
                        logger.info(f"转录文本已保存: {transcript_path}")
                
                elif data_key == "metadata":
                    # 保存元数据到JSON文件
                    metadata_path = work_dir / "metadata.json"
                    metadata_copy = data.copy()
                    with open(metadata_path, "w", encoding="utf-8") as f:
                        json.dump(metadata_copy, f, indent=2, ensure_ascii=False, default=str)
                    data["metadata_file"] = str(metadata_path)
                    logger.info(f"元数据已保存: {metadata_path}")
                
                elif data_key == "motion_tokens":
                    # 保存运动令牌到文件
                    tokens_path = work_dir / "motion_tokens.json"
                    with open(tokens_path, "w", encoding="utf-8") as f:
                        json.dump(data, f, indent=2, ensure_ascii=False)
                    data["motion_tokens_file"] = str(tokens_path)
                    logger.info(f"运动令牌已保存: {tokens_path}")

                elif data_key == "video_tokens":
                    # 保存视频tokens到文件，确保跨请求可复用
                    tokens_path = work_dir / "video_tokens.json"
                    with open(tokens_path, "w", encoding="utf-8") as f:
                        json.dump(data, f, indent=2, ensure_ascii=False)
                    data["video_tokens_file"] = str(tokens_path)
                    logger.info(f"视频tokens已保存: {tokens_path}")
                
                # 保存到内存中的会话数据
                self.sessions[session_id]["processed_data"][data_key] = data
                
                # 更新会话信息文件
                self._update_session_info_file(session_id)
                
                logger.info(f"处理数据已保存: {session_id} -> {data_key}")
                return True
                
            except Exception as e:
                logger.error(f"保存处理数据时出错 {session_id}/{data_key}: {e}")
                return False

    def _update_session_info_file(self, session_id: str):
        """更新会话信息文件"""
        try:
            session_data = self.sessions[session_id]
            work_dir = Path(session_data["work_dir"])
            session_info_file = work_dir / "session_info.json"
            
            session_info = {
                "session_id": session_id,
                "created_at": session_data["created_at"].isoformat(),
                "last_updated": datetime.now().isoformat(),
                "original_filename": session_data["original_file"]["filename"],
                "file_size": session_data["original_file"]["size"],
                "processed_steps": list(session_data["processed_data"].keys())
            }
            
            with open(session_info_file, "w", encoding="utf-8") as f:
                json.dump(session_info, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"更新会话信息文件失败 {session_id}: {e}")
    

    def save_file_to_session_dir(self, session_id: str, file_data: str, filename: str, subdir: str = "") -> str:
        """将文件保存到会话目录中的指定子目录"""
        work_dir = self.get_session_work_dir(session_id)
        if not work_dir:
            raise ValueError(f"会话不存在: {session_id}")
        
        target_dir = work_dir / subdir if subdir else work_dir
        target_dir.mkdir(parents=True, exist_ok=True)
        
        file_path = target_dir / filename
        with open(file_path, "wb") as f:
            f.write(base64.b64decode(file_data))
        
        logger.info(f"文件保存到会话目录: {file_path}")
        return str(file_path)
    

    def save_temp_file(self, file_data: str, extension: str, session_id: str = None) -> str:
        """保存临时文件, 优先保存到会话的temp目录"""
        try:
            if session_id:
                work_dir = self.get_session_work_dir(session_id)
                if work_dir:
                    temp_dir = work_dir / "temp"
                    temp_dir.mkdir(exist_ok=True)
                    file_id = str(uuid.uuid4())[:8]
                    file_path = temp_dir / f"temp_{file_id}.{extension.lstrip('.')}"
                else:
                    # 回退到全局临时目录
                    file_id = str(uuid.uuid4())
                    file_path = self.base_work_dir / f"global_temp_{file_id}.{extension.lstrip('.')}"
            else:
                # 全局临时文件
                file_id = str(uuid.uuid4())
                file_path = self.base_work_dir / f"global_temp_{file_id}.{extension.lstrip('.')}"
            
            # 解码并验证 base64 数据
            decoded_data = base64.b64decode(file_data)
            
            with open(file_path, "wb") as f:
                f.write(decoded_data)
            
            logger.debug(f"保存临时文件: {file_path} ({len(decoded_data)} bytes)")
            return str(file_path)
        except Exception as e:
            logger.error(f"保存临时文件失败: {e}")
            raise

    def encode_file_to_base64(self, file_path: str) -> str:
        """将文件编码为 base64"""
        try:
            with open(file_path, "rb") as f:
                data = f.read()
                return base64.b64encode(data).decode('utf-8')
        except Exception as e:
            logger.error(f"编码文件失败 {file_path}: {e}")
            raise
    
    def encode_files_to_base64(self, file_paths: list) -> list:
        """将多个文件编码为 base64"""
        result = []
        for file_path in file_paths:
            try:
                if os.path.exists(file_path):
                    result.append({
                        "filename": os.path.basename(file_path),
                        "data": self.encode_file_to_base64(file_path),
                        "size": os.path.getsize(file_path),
                        "path": file_path  # 服务器端路径信息
                    })
            except Exception as e:
                logger.error(f"编码文件失败 {file_path}: {e}")
        return result
    
    def cleanup_temp_files(self, *file_paths):
        """清理临时文件"""
        for file_path in file_paths:
            try:
                if file_path and os.path.exists(file_path):
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                        logger.debug(f"删除临时文件: {file_path}")
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                        logger.debug(f"删除临时目录: {file_path}")
            except Exception as e:
                logger.error(f"清理文件失败 {file_path}: {e}")
    
    def cleanup_session(self, session_id: str):
        """清理整个会话及其目录"""
        with self.lock:
            if session_id not in self.sessions:
                logger.warning(f"尝试清理不存在的会话: {session_id}")
                return
            
            session_data = self.sessions[session_id]
            work_dir = session_data.get("work_dir")
            
            # 删除整个会话工作目录
            if work_dir and os.path.exists(work_dir):
                try:
                    shutil.rmtree(work_dir)
                    logger.info(f"删除会话目录: {work_dir}")
                except Exception as e:
                    logger.error(f"删除会话目录失败 {work_dir}: {e}")
            
            # 删除会话记录
            del self.sessions[session_id]
            logger.info(f"清理会话完成: {session_id}")
    
    def get_session_stats(self) -> Dict:
        """获取会话统计信息"""
        with self.lock:
            total_sessions = len(self.sessions)
            total_size = sum(
                session["original_file"]["size"] 
                for session in self.sessions.values()
            )
            
            # 计算总的磁盘使用
            total_disk_usage = 0
            for session_data in self.sessions.values():
                work_dir = session_data.get("work_dir")
                if work_dir and os.path.exists(work_dir):
                    for dirpath, dirnames, filenames in os.walk(work_dir):
                        for filename in filenames:
                            filepath = os.path.join(dirpath, filename)
                            if os.path.exists(filepath):
                                total_disk_usage += os.path.getsize(filepath)
            
            return {
                "total_sessions": total_sessions,
                "total_original_data_size": total_size,
                "total_disk_usage": total_disk_usage,
                "total_disk_usage_mb": round(total_disk_usage / (1024*1024), 2),
                "oldest_session": min(
                    (session["created_at"] for session in self.sessions.values()),
                    default=datetime.now()
                ).isoformat() if self.sessions else None,
                "session_list": list(self.sessions.keys())
            }
    
    def list_session_files(self, session_id: str) -> Dict:
        """列出会话目录中的所有文件"""
        try:
            work_dir = self.get_session_work_dir(session_id)
            if not work_dir or not work_dir.exists():
                return {"status": "error", "message": "会话目录不存在"}
            
            file_structure = {}
            total_files = 0
            total_size = 0
            
            for root, dirs, files in os.walk(work_dir):
                rel_path = os.path.relpath(root, work_dir)
                if rel_path == ".":
                    rel_path = "root"
                
                file_list = []
                for f in files:
                    file_path = os.path.join(root, f)
                    file_size = os.path.getsize(file_path)
                    total_files += 1
                    total_size += file_size
                    
                    file_list.append({
                        "name": f,
                        "size": file_size,
                        "size_mb": round(file_size / (1024*1024), 3),
                        "modified": datetime.fromtimestamp(
                            os.path.getmtime(file_path)
                        ).isoformat()
                    })
                
                file_structure[rel_path] = {
                    "directories": dirs,
                    "files": file_list,
                    "file_count": len(file_list)
                }
            
            return {
                "status": "success",
                "session_id": session_id,
                "work_directory": str(work_dir),
                "total_files": total_files,
                "total_size": total_size,
                "total_size_mb": round(total_size / (1024*1024), 2),
                "file_structure": file_structure
            }
        except Exception as e:
            logger.error(f"列出会话文件时出错: {e}")
            return {"status": "error", "message": f"列出会话文件时出错: {str(e)}"}
    