# backend/Dockerfile

# Use a specific, slim, and secure base image.
FROM python:3.10-slim

# --- 1. DEFINE USER AND ENVIRONMENT ---
ARG APP_USER=app
ARG APP_HOME=/home/app
ENV HOME=${APP_HOME}
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
# Set the PATH to include the user's local bin directory
ENV PATH=${APP_HOME}/.local/bin:${PATH}

# --- 2. CREATE USER, DIRECTORIES, AND INSTALL SYSTEM DEPS (as root) ---
RUN addgroup --system ${APP_USER} && adduser --system --group ${APP_USER} \
    && mkdir -p ${APP_HOME}/.config \
    && chown ${APP_USER}:${APP_USER} ${APP_HOME}/.config

RUN apt-get update && apt-get install -y --no-install-recommends --no-install-suggests curl gnupg \
    && curl -sSL https://ngrok-agent.s3.amazonaws.com/ngrok.asc | tee /etc/apt/trusted.gpg.d/ngrok.asc >/dev/null \
    && echo "deb https://ngrok-agent.s3.amazonaws.com buster main" | tee /etc/apt/sources.list.d/ngrok.list \
    && apt-get update \
    && apt-get install -y ngrok \
    && apt-get purge -y curl gnupg && apt-get autoremove -y && rm -rf /var/lib/apt/lists/*

# --- 3. SWITCH TO NON-ROOT USER FOR DEPENDENCY INSTALLATION ---
WORKDIR ${APP_HOME}

# Copy and install Python dependencies as the non-root user.
# This prevents root-owned files in the user's home directory.
COPY --chown=${APP_USER}:${APP_USER} setup.py .
RUN pip install --no-cache-dir --user .

# --- 4. COPY APPLICATION CODE ---
COPY --chown=${APP_USER}:${APP_USER} ./app ./app
COPY --chown=${APP_USER}:${APP_USER} run.py .
RUN chown -R ${APP_USER}:${APP_GROUP} ${APP_HOME}

# --- 5. RUN THE APPLICATION ---
EXPOSE 5000
USER ${APP_USER}
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "run:app"]