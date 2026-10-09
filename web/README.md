# 🌐 Archaeologist Web Dashboard (Experimental)

This directory contains the standalone frontend assets for the Archaeologist web interface.

> ℹ️ **Archaeologist CLI** is primarily designed as a terminal-first forensic intelligence tool. This web dashboard is an optional graphical interface for visualizing force-directed causal graphs in the browser.

---

## Running the Web UI

To launch the web dashboard server:

```bash
archaeologist ui
```

Or run headless:

```bash
archaeologist serve --port 8000
```

Then navigate to `http://127.0.0.1:8000` in your browser.

---

## Assets
- `index.html`: Main dashboard template.
- `app.js`: Force-directed graph rendering and streaming query client.
- `index.css`: Glassmorphic dark theme stylesheet.
- `launch-video.html`: Animated showcase presentation.
