"""Setup script for AI Call MVP."""

from setuptools import setup, find_packages

setup(
    name="ai-call-mvp",
    version="0.1.0",
    description="AI-powered call handling MVP",
    author="Abhay Kashyap",
    author_email="abhay.kashyap03@gmail.com",
    packages=find_packages(),
    install_requires=[
        "flask",
        "twilio",
        "openai",
        "google-generativeai",
        "sentence-transformers",
        "faiss-cpu",
        "beautifulsoup4",
        "pydub",
        "whisper",
        "python-dotenv",
        "pytest",
        "ngrok",
        "requests",
        "markdown",
    ],
    entry_points={
        "console_scripts": [
            "ai-call-cli=cli:main",
        ],
    },
    python_requires=">=3.8",
)
