FROM python:3.11

WORKDIR /app

COPY . /app

RUN pip install -e . --find-links https://data.pyg.org/whl/torch-2.1.2+cpu.html

EXPOSE 8002

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8002"]