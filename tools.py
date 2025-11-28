import os
import shutil
import json
import subprocess
from pathlib import Path
from typing import Dict, Optional
from langchain.tools import tool
from pydantic.v1 import BaseModel, Field

VIDEO_ROOT = os.getenv("VIDEO_ROOT", "/video")
# Define root directory for operations
ROOT_DIR = Path(VIDEO_ROOT).resolve()

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
    srt_filename: str = Field(..., description="Name of the subtitle file in tmp dir to sync")
    output_srt_name: str = Field(..., description="Name for the synced output subtitle file. Identical to the video name, but with a `_synced` suffix appended.")

@tool("sync_subtitles", args_schema=SyncSubtitleArgs)
def sync_subtitles(video_filename: str, srt_filename: str, output_srt_name: str) -> Dict:
    """
    Synchronize subtitles using ffsubsync. Both video and subtitle must be in the temporary directory.
    """
    try:
        tmp_dir = ROOT_DIR / "tmp"
        video_path = tmp_dir / video_filename
        srt_path = tmp_dir / srt_filename
        output_path = tmp_dir / output_srt_name

        if not video_path.exists():
            return {"status": "error", "message": f"Reference video not found at {video_filename}"}
        if not srt_path.exists():
            return {"status": "error", "message": f"Subtitle file not found at {srt_filename}"}

        cmd = ["ffs", str(video_path), "-i", str(srt_path), "-o", str(output_path)]
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
    Check if there is an external subtitle file (same name as video) in the same directory.
    Returns the path to the subtitle if found.
    """
    try:
        target_path = get_safe_path(video_path)
        base_name = target_path.stem
        parent_dir = target_path.parent
        
        for ext in ['.srt', '.ass', '.vtt']:
            sub_path = parent_dir / (base_name + ext)
            if sub_path.exists():
                return {"status": "success", "subtitle_path": str(sub_path.relative_to(ROOT_DIR))}
        
        return {"status": "success", "subtitle_path": None}
    except Exception as e:
        return {"status": "error", "message": f"Error checking subtitle: {str(e)}"}

class CleanupSubtitleArgs(BaseModel):
    temp_dir: str = Field(default="tmp", description="The temporary directory path relative to ROOT_DIR")

@tool("cleanup_subtitle", args_schema=CleanupSubtitleArgs)
def cleanup_subtitle(temp_dir: str = "tmp") -> Dict:
    """
    Clean up non-synced subtitle files and rename synced subtitle files in the temporary directory:
    1. Delete all non-synced subtitle files (files without _sync suffix)
    2. Rename synced subtitle files by removing the _sync suffix to match video names
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
            if '_sync' in filename:
                # Remove _sync suffix to match video name
                new_filename = filename.replace('_sync', '')
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
