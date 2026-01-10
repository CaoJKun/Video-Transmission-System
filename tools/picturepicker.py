import base64
import uuid
import json
import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

async def picturepicker(
    session_id: str = None,
    frames_data: list = None,
    max_keyframes: int = 8,
    selection_method: str = "uniform",
    tools_manager = None
) -> dict:
    """从视频帧中选择关键帧 - 远程服务器版本，既保存又返回"""
    try:
        # 验证参数
        max_keyframes = max(1, min(max_keyframes, 50))
        if selection_method not in ["uniform", "random"]:
            selection_method = "uniform"
        
        # 获取帧数据
        if session_id:
            frames_info = tools_manager.get_session_data(session_id, "frames")
            if not frames_info:
                return {"status": "error", "message": "未找到视频帧数据，请先调用 video2pictures"}
            
            encoded_frames = frames_info["encoded_frames"]
            work_dir = tools_manager.get_session_work_dir(session_id)
            keyframes_dir = work_dir / "keyframes"
            keyframes_dir.mkdir(exist_ok=True)
            # 清理旧的关键帧文件
            if keyframes_dir.exists():
                for old_file in keyframes_dir.glob("keyframe_*.jpg"):
                    try:
                        old_file.unlink()
                    except Exception as e:
                        logger.warning(f"清理旧关键帧文件失败: {e}")
        elif frames_data:
            encoded_frames = frames_data
            keyframes_dir = None
        else:
            return {"status": "error", "message": "需要提供 session_id 或 frames_data"}
        
        if not encoded_frames:
            return {"status": "error", "message": "没有可用的帧数据"}
        
        # 选择关键帧
        total = len(encoded_frames)
        max_keyframes = min(max_keyframes, total)
        
        if selection_method == "uniform":
            if max_keyframes == 1:
                selected_indices = [total // 2]
            else:
                # 提高取帧频率：使用更小的固定步长，提取更多关键帧
                # 每5帧取一帧，确保提取足够多的关键帧
                step = 5  # 固定步长为5，每5帧取一帧
                selected_indices = list(range(0, total, step))
                logger.info(f"提取关键帧: 总帧数={total}, 步长={step}, 将提取{len(selected_indices)}帧")
        else:  # random
            import random
            selected_indices = sorted(random.sample(range(total), max_keyframes))
        
        # 获取选择的关键帧
        selected_frames = []
        saved_count = 0
        for i, idx in enumerate(selected_indices):
            if idx < len(encoded_frames):
                frame_data = encoded_frames[idx].copy()
                frame_data["keyframe_index"] = i
                frame_data["original_index"] = idx
                selected_frames.append(frame_data)
                
                # 如果有会话目录，也保存关键帧到服务器
                if keyframes_dir and "data" in frame_data:
                    keyframe_path = keyframes_dir / f"keyframe_{i:04d}.jpg"
                    try:
                        with open(keyframe_path, "wb") as f:
                            f.write(base64.b64decode(frame_data["data"]))
                        frame_data["server_path"] = str(keyframe_path)
                        saved_count += 1
                    except Exception as e:
                        logger.error(f"保存关键帧到服务器失败: {e}")
                elif keyframes_dir:
                    logger.warning(f"关键帧 {i} (索引 {idx}) 没有 data 字段，跳过保存")
        
        logger.info(f"关键帧选择完成: 提取{len(selected_frames)}张, 保存{saved_count}张到服务器")
        
        # 准备关键帧数据
        keyframes_data = {
            "selected_frames": selected_frames,
            "selected_indices": selected_indices,
            "selection_method": selection_method,
            "total_original_frames": total,
            "keyframes_dir": str(keyframes_dir) if keyframes_dir else None
        }
        
        # 如果有会话，保存关键帧结果到服务器
        if session_id:
            tools_manager.save_processed_data(session_id, "keyframes", keyframes_data)
            logger.info(f"关键帧已保存到服务器: {keyframes_dir}")
        
        # 返回给客户端的完整数据
        return {
            "status": "success",
            "session_id": session_id,
            "total_frames": total,
            "selected_keyframes": len(selected_frames),
            "selection_method": selection_method,
            "keyframes": selected_frames,  # 完整关键帧数据返回给客户端
            "server_info": {
                "keyframes_saved_to": str(keyframes_dir) if keyframes_dir else None,
                "selection_completed_at": datetime.now().isoformat()
            }
        }
    
    except Exception as e:
        logger.error(f"选择关键帧时出错: {e}")
        return {"status": "error", "message": f"选择关键帧时出错: {str(e)}"}
