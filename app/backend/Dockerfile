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
    && apt-get purge -y cur