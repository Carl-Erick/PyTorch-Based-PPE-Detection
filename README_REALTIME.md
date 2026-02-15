Real-time SSE demo

- Start the Flask SSE server (will serve `static/index.html` and `/stream`):

```bash
python3 app.py   # or use the VS Code launch: "Run Flask app.py (port 8001)"
```

- By default the app reads `PORT` env; the VS Code launch uses `8001`. To run on 8001 manually:

```bash
PORT=8001 python3 app.py
```

- Open in browser: http://localhost:8001 — the page connects to `/stream` and renders a live line chart.

- Stop the background process (if started in background):

```bash
kill <pid>
```

Notes:
- Dependencies: See `Codebase_Sprint1/requirements_simple.txt` (Flask added).
- The client uses Chart.js from CDN; no frontend build needed.
