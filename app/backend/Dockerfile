FROM python:3.10.13-slim-bookworm

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# --- Paths are relative to the build context (project root) ---
COPY setup.py .
RUN pip install --no-cache-dir .

# Copy your application code
COPY ./app/backend ./app/backend
COPY ./app/__init__.py ./app/__init__.py
COPY run.py .
COPY settings.json .

RUN addgroup --system app && adduser --system --group app
# Change ownership of the app directory inside the container
RUN chown -R app:app /app
USER app

EXPOSE 5000

CMD ["gunicorn", "--bind", "0.0.0:5000", "run:app"]