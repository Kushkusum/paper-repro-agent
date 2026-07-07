FROM python:3.11-slim

RUN pip install --no-cache-dir numpy scipy

WORKDIR /work

# Code is mounted at runtime; no COPY needed. Run as non-root for defense in depth.
RUN useradd -m runner
USER runner

CMD ["python", "run.py"]
