import os
import shutil
import json
import subprocess
from pathlib import Path
from typing import Dict, Optional
from langchain.tools import tool
from pydantic.v1 import BaseModel, Field
from langchain_openai import ChatOpenAI

# Define root directory for video
ROOT_DIR = Path("/video").resolve()

# Initialize LLM for smart subtitle matching
api_key = os.getenv("OPENAI_API_KEY", "")
base_url = os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com/v1")
model = os.getenv("MODEL_NAME", "deepseek-chat")
llm = ChatOpenAI(model=model, temperature=0, api_key=api_key, base_url=base_url)

def get_safe_path(user_path: str) -> Path:
    """Ensure the path is within the allowed ROOT_DIR."""
    if not user_path or user_path == ".":
        return ROOT_DIR
    
    # Handle absolute paths that might be passed by the LLM
    if user_path.startswith(str(ROOT_DIR)):
        try:
            target_path = Path(user_path).resolve()
        except Exception:
             # If resolving fails, treat as relative
             clean_path = user_path.lstrip("/")
             target_path = (ROOT_DIR / clean_path).resolve()
    else:
        clean_path = user_path.lstrip("/")
        target_path = (ROOT_DIR / clean_path).resolve()

    # Strict security check: must be inside ROOT_DIR
    if not str(target_path).startswith(str(ROOT_DIR)):
        raise ValueError(f"Access denied: Cannot access {target_path}. Only {ROOT_DIR} is allowed.")

    return target_path

def run_command(cmd_list):
    """Execute a shell command and return stdout/stderr."""
    try:
        result = subprocess.run(
            cmd_list,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        return {"status": "success", "stdout": result.stdout, "stderr": result.stderr}
    except subprocess.CalledProcessError as e:
        return {"status": "error", "message": str(e), "stderr": e.stderr}

class ListVideoArgs(BaseModel):
    query: Optional[str] = Field(None, description="Optional search query to filter file names")

@tool("list_video_files", args_schema=ListVideoArgs)
def list_video_files(query: Optional[str] = None) -> Dict:
    """
    Recursively list all video files in the root directory. 
    Returns a dictionary with status and a list of file paths.
    """
    video_extensions = {'.mp4', '.mkv', '.avi', '.mov', '.flv', '.wmv'}
    found_files = []
    
    try:
        for root, _, files in os.walk(ROOT_DIR):
            for file in files:
                if Path(file).suffix.lower() in video_extensions:
                    full_path = Path(root) / file
                    relative_path = str(full_path.relative_to(ROOT_DIR))
                    
                    if query:
                        if query.lower() in file.lower():
                            found_files.append(relative_path)
                    else:
                        found_files.append(relative_path)
        
        return {"status": "success", "files": found_files}
    except Exception as e:
        return {"status": "error", "message": f"Error listing files: {str(e)}"}

class FilePathArgs(BaseModel):
    file_path: str = Field(..., description="The relative path to the video file (e.g. 'movie/test.mp4')")

@tool("get_video_info", args_schema=FilePathArgs)
def get_video_info(file_path: str) -> Dict:
    """
    Get stream information for a specific video file using ffprobe.
    Returns a dictionary with status, format, and separated lists for video, audio, and subtitle streams.
    """
    try:
        target_path = get_safe_path(file_path)
        if not target_path.is_file():
            return {"status": "error", "message": f"File not found at {file_path}"}

        cmd = [
            "ffprobe", "-v", "quiet", "-print_format", "json",
            "-show_streams", "-show_format", str(target_path)
        ]
        res = run_command(cmd)
        
        if res["status"] == "success":
            raw_data = json.loads(res["stdout"])
            all_streams = raw_data.get("streams", [])
            
            # 分类筛选
            video_streams = [s for s in all_streams if s.get("codec_type") == "video"]
            audio_streams = [s for s in all_streams if s.get("codec_type") == "audio"]
            subtitle_streams = [s for s in all_streams if s.get("codec_type") == "subtitle"]
            
            return {
                "status": "success",
                "format": raw_data.get("format", {}),
                "video": video_streams,
                "audio": audio_streams,
                "subtitles": subtitle_streams
            }
            
        return {"status": "error", "message": f"Error running ffprobe: {res.get('stderr')}"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

class ExtractStreamsArgs(BaseModel):
    input_path: str = Field(..., description="Relative path to source video")
    output_filename: str = Field(..., description="Name for the extracted file, identical to the input file name")
    video_stream_index: int = Field(..., description="Index of the video stream to copy")
    audio_stream_index: int = Field(..., description="Index of the audio stream to copy")

@tool("extract_video_audio", args_schema=ExtractStreamsArgs)
def extract_video_audio(input_path: str, output_filename: str, video_stream_index: int, audio_stream_index: int) -> Dict:
    """
    Extract specific video and audio streams to a new file in the temporary directory.
    """
    try:
        source = get_safe_path(input_path)
        tmp_dir = ROOT_DIR / "tmp"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        dest = tmp_dir / output_filename

        cmd = ["ffmpeg", "-y", "-i", str(source),
               "-map", f"0:{video_stream_index}",
               "-map", f"0:{audio_stream_index}",
               "-c", "copy", str(dest)]

        res = run_command(cmd)
        if res["status"] == "success":
            return {"status": "success", "message": f"Successfully extracted streams to {dest.relative_to(ROOT_DIR)}"}
        return {"status": "error", "message": f"Error extracting streams: {res.get('stderr')}"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

class ExtractSubtitleArgs(BaseModel):
    input_path: str = Field(..., description="Relative path to source video")
    output_filename: str = Field(..., description="Name for the extracted subtitle file (e.g. 'movie_name.srt'), identical to the source video")
    stream_index: int = Field(0, description="Index of the subtitle stream")

@tool("extract_subtitle", args_schema=ExtractSubtitleArgs)
def extract_subtitle(input_path: str, output_filename: str, stream_index: int = 0) -> Dict:
    """
    Extract a subtitle stream from a video file to the temporary directory.
    """
    try:
        source = get_safe_path(input_path)
        tmp_dir = ROOT_DIR / "tmp"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        dest = tmp_dir / output_filename

        cmd = ["ffmpeg", "-y", "-i", str(source), "-map", f"0:{stream_index}", str(dest)]

        res = run_command(cmd)
        if res["status"] == "success":
            return {"status": "success", "message": f"Successfully extracted subtitle to {dest.relative_to(ROOT_DIR)}"}
        return {"status": "error", "message": f"Error extracting subtitle: {res.get('stderr')}"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

class SyncSubtitleArgs(BaseModel):
    video_filename: str = Field(..., description="Name of the reference video file in tmp dir")
    subtitle_filename: str = Field(..., description="Name of the subtitle file in tmp dir to sync")
    output_subtitle_name: str = Field(..., description="Name for the synced output subtitle file. Identical to the video name, but with a `_synced` suffix appended.")

@tool("sync_subtitles", args_schema=SyncSubtitleArgs)
def sync_subtitles(video_filename: str, subtitle_filename: str, output_subtitle_name: str) -> Dict:
    """
    Synchronize subtitles using ffsubsync. Both video and subtitle must be in the temporary directory.
    """
    try:
        tmp_dir = ROOT_DIR / "tmp"
        video_path = tmp_dir / video_filename
        subtitle_path = tmp_dir / subtitle_filename
        
        original_ext = Path(subtitle_filename).suffix.lower()
        output_path = tmp_dir / (Path(output_subtitle_name).stem + original_ext)

        if not video_path.exists():
            return {"status": "error", "message": f"Reference video not found at {video_filename}"}
        if not subtitle_path.exists():
            return {"status": "error", "message": f"Subtitle file not found at {subtitle_filename}"}

        cmd = ["ffs", str(video_path), "-i", str(subtitle_path), "-o", str(output_path)]
        res = run_command(cmd)

        if res["status"] == "success":
            return {"status": "success", "message": f"Successfully synced subtitles. Output saved to {output_path.relative_to(ROOT_DIR)}"}
        return {"status": "error", "message": f"Error syncing subtitles: {res.get('stderr')}"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@tool("cleanup_temp")
def cleanup_temp() -> Dict:
    """
    Delete all files in the temporary directory.
    """
    try:
        tmp_path = ROOT_DIR / "tmp"
        if tmp_path.exists():
            shutil.rmtree(tmp_path)
            tmp_path.mkdir(exist_ok=True)
            return {"status": "success", "message": "Temporary directory cleaned successfully."}
        return {"status": "success", "message": "Temporary directory does not exist."}
    except Exception as e:
        return {"status": "error", "message": f"Error cleaning temp directory: {str(e)}"}

@tool("list_temp_files")
def list_temp_files() -> Dict:
    """
    List all files currently in the temporary directory.
    """
    try:
        tmp_path = ROOT_DIR / "tmp"
        if not tmp_path.exists():
            return {"status": "success", "files": []}
        
        files = [f for f in os.listdir(tmp_path) if (tmp_path / f).is_file()]
        return {"status": "success", "files": files}
    except Exception as e:
        return {"status": "error", "message": f"Error listing temp files: {str(e)}"}

class CheckSubtitleArgs(BaseModel):
    video_path: str = Field(..., description="Relative path to the video file")

@tool("copy_to_temp", args_schema=FilePathArgs)
def copy_to_temp(file_path: str) -> Dict:
    """
    Copy a file from the main video storage to the temporary directory.
    """
    try:
        source = get_safe_path(file_path)
        if not source.is_file():
            return {"status": "error", "message": f"Source file not found at {file_path}"}

        tmp_dir = ROOT_DIR / "tmp"
        tmp_dir.mkdir(parents=True, exist_ok=True)

        dest = tmp_dir / source.name
        shutil.copy2(source, dest)

        return {"status": "success", "message": f"Successfully copied {source.name} to temporary directory."}
    except Exception as e:
        return {"status": "error", "message": f"Error copying file: {str(e)}"}

@tool("check_external_subtitle", args_schema=CheckSubtitleArgs)
def check_external_subtitle(video_path: str) -> Dict:
    """
    Check for external subtitle files that match the video semantically using LLM.
    Returns matching subtitle file paths in JSON format.
    Only returns subtitles that are semantically the same video.
    
    Returns:
        {
            "status": "success" or "error",
            "subtitle_paths": ["path/to/subtitle1.srt", "path/to/subtitle2.srt"] or null,
            "count": number of matched subtitles (if success)
        }
    """
    try:
        target_path = get_safe_path(video_path)
        base_name = target_path.stem
        parent_dir = target_path.parent
        
        # Step 1: Exact match first (check for exact filename matches)
        exact_matches = []
        for ext in ['.srt', '.ass', '.vtt']:
            sub_path = parent_dir / (base_name + ext)
            if sub_path.exists():
                exact_matches.append(str(sub_path.relative_to(ROOT_DIR)))
        
        if exact_matches:
            return {
                "status": "success",
                "subtitle_paths": exact_matches,
                "count": len(exact_matches),
                "match_type": "exact"
            }
        
        # Step 2: Collect all subtitle candidates in the same directory
        subtitle_extensions = {'.srt', '.ass', '.vtt'}
        candidates = []
        if parent_dir.exists():
            for f in parent_dir.iterdir():
                if f.is_file() and f.suffix.lower() in subtitle_extensions:
                    candidates.append((f.stem, str(f.relative_to(ROOT_DIR)), f))
        
        if not candidates:
            return {
                "status": "success",
                "subtitle_paths": None,
                "count": 0,
                "match_type": "none"
            }
        
        # Step 3: Use LLM for semantic matching
        candidate_info = [{"name": name, "path": rel_path} for name, rel_path, _ in candidates]
        
        # Enhanced prompt to ensure we get all semantically matching subtitles
        prompt = f"""分析视频与字幕文件的对应关系。
视频名称: {base_name}
候选字幕文件列表: {json.dumps(candidate_info, ensure_ascii=False)}

任务: 找出所有在语义上属于同一个视频的字幕文件。
- 如果文件名明显是同一部视频的不同语言/版本字幕，全部返回
- 只返回确实对应该视频的字幕文件
- 如果没有匹配的字幕，返回"NONE"

返回格式: 每行一个匹配的字幕文件路径(如: path/subtitle.srt),或返回"NONE"
"""
        
        response = llm.invoke(prompt)
        response_text = response.content.strip()
        
        # Parse LLM response
        if response_text.upper() == "NONE":
            return {
                "status": "success",
                "subtitle_paths": None,
                "count": 0,
                "match_type": "llm_no_match"
            }
        
        # Extract matched subtitle paths from response
        matched_paths = []
        for line in response_text.split('\n'):
            cleaned = line.strip()
            if cleaned and cleaned.upper() != "NONE":
                matched_paths.append(cleaned)
        
        # Validate paths exist in candidates
        matched_files = []
        candidate_paths = {rel_path: rel_path for _, rel_path, _ in candidates}
        for path in matched_paths:
            if path in candidate_paths:
                matched_files.append(path)
        
        if matched_files:
            return {
                "status": "success",
                "subtitle_paths": matched_files,
                "count": len(matched_files),
                "match_type": "llm"
            }
        
        # No matches found
        return {
            "status": "success",
            "subtitle_paths": None,
            "count": 0,
            "match_type": "llm_no_match"
        }
            
    except Exception as e:
        return {
            "status": "error",
            "subtitle_paths": None,
            "count": 0,
            "message": str(e)
        }

class CleanupSubtitleArgs(BaseModel):
    temp_dir: str = Field(default="tmp", description="The temporary directory path relative to ROOT_DIR")

@tool("cleanup_subtitle", args_schema=CleanupSubtitleArgs)
def cleanup_subtitle(temp_dir: str = "tmp") -> Dict:
    """
    Clean up non-synced subtitle files and rename synced subtitle files in the temporary directory:
    1. Delete all non-synced subtitle files (files without _synced suffix)
    2. Rename synced subtitle files by removing the _synced suffix to match video names
    Returns a summary of operations performed.
    """
    try:
        tmp_path = ROOT_DIR / temp_dir
        if not tmp_path.exists():
            return {"status": "success", "message": "Temporary directory does not exist.", "deleted": [], "renamed": []}
        
        deleted_files = []
        renamed_files = []
        
        # Get all subtitle files in the directory
        subtitle_extensions = {'.srt', '.ass', '.vtt'}
        files = [f for f in os.listdir(tmp_path) if (tmp_path / f).is_file()]
        
        for filename in files:
            file_path = tmp_path / filename
            file_ext = Path(filename).suffix.lower()
            
            # Only process subtitle files
            if file_ext not in subtitle_extensions:
                continue
            
            # Check if this is a synced subtitle file
            if '_synced' in filename:
                # Remove _synced suffix to match video name
                new_filename = filename.replace('_synced', '')
                new_path = tmp_path / new_filename
                
                try:
                    if new_path.exists():
                        # If target already exists, remove it first
                        new_path.unlink()
                    file_path.rename(new_path)
                    renamed_files.append({
                        "old_name": filename,
                        "new_name": new_filename
                    })
                except Exception as e:
                    return {"status": "error", "message": f"Error renaming {filename}: {str(e)}"}
            else:
                # This is a non-synced subtitle file, delete it
                try:
                    file_path.unlink()
                    deleted_files.append(filename)
                except Exception as e:
                    return {"status": "error", "message": f"Error deleting {filename}: {str(e)}"}
        
        summary = f"Successfully cleaned up subtitle files. "
        if deleted_files:
            summary += f"Deleted {len(deleted_files)} non-synced file(s): {', '.join(deleted_files)}. "
        if renamed_files:
            summary += f"Renamed {len(renamed_files)} synced file(s): {', '.join([r['old_name'] + ' → ' + r['new_name'] for r in renamed_files])}."
        
        return {
            "status": "success",
            "message": summary,
            "deleted": deleted_files,
            "renamed": renamed_files
        }
    except Exception as e:
        return {"status": "error", "message": f"Error cleaning up subtitles: {str(e)}"}
