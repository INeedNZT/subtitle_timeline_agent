FROM python:3.11-slim

# 安装系统依赖，阿里云源
RUN sed -i 's/deb.debian.org/mirrors.aliyun.com/g' /etc/apt/sources.list.d/debian.sources && \
    apt-get update && \
    apt-get install -y ffmpeg gcc && \
    rm -rf /var/lib/apt/lists/*

# 复制项目文件
COPY requirements.txt .
COPY app.py .
COPY tools.py .

# 安装Python依赖，清华源
RUN pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple 

# 暴露端口
EXPOSE 80

# 启动应用
CMD ["python", "app.py"]
