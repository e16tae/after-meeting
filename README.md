# After Meeting

AI-powered meeting audio to minutes/report generator.

## Installation

```bash
uv pip install -e ".[dev]"
```

## Usage

```bash
# Full pipeline
after-meeting process meeting.mp3 --doc-type minutes --format docx --json

# Step-by-step
after-meeting transcribe meeting.mp3 --json
after-meeting structure transcript.json --doc-type minutes --json
after-meeting render structured.json --format docx --json
```
