FROM pytorch/pytorch:2.4.1-cuda12.4-cudnn9-runtime

WORKDIR /app
ENV PYTHONPATH=/app/src:/app/src/agewell/vendor

COPY src/agewell/services/brainiac_svc/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY src /app/src
COPY models/licenses /app/models/licenses

CMD ["uvicorn", "agewell.services.brainiac_svc.api:app", "--host", "0.0.0.0", "--port", "8000"]
