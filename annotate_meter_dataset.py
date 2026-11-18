"""Create verified digit crops from the archived meter dataset using a local browser UI."""
from __future__ import annotations

import argparse
import base64
import csv
import json
import math
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse
import webbrowser

import cv2

from sevenseg.classifier import normalise_crop


QUALITIES = ("HQ_digital", "MQ_digital", "LQ_digital")


@dataclass(frozen=True)
class AnnotationTask:
    quality: str
    filename: str
    labels: str
    image_path: Path

    @property
    def task_id(self) -> str:
        return f"{self.quality}:{self.filename}"


def _label_from_row(row: dict[str, str]) -> str:
    return "".join(
        value.strip()
        for index in range(1, 5)
        if (value := (row.get(f"cadran_{index}") or "")).strip().isdigit()
    )


def _crop_path(output_dir: Path, task: AnnotationTask, position: int) -> Path:
    stem = Path(task.filename).stem
    return output_dir / task.labels[position] / f"{task.quality}_{stem}_{position}.png"


def load_tasks(root: Path, output_dir: Path, qualities: list[str]) -> tuple[list[AnnotationTask], int]:
    tasks = []
    skipped = 0
    for quality in qualities:
        csv_path = root / f"{quality}.csv"
        image_dir = root / quality
        if not csv_path.is_file():
            raise FileNotFoundError(f"Dataset labels not found: {csv_path}")
        with csv_path.open(newline="", encoding="utf-8-sig") as source:
            for row in csv.DictReader(source, delimiter=";"):
                filename = (row.get("image") or "").strip()
                labels = _label_from_row(row)
                reading = "".join(c for c in (row.get("used_liter") or "") if c.isdigit())
                if (not filename or Path(filename).name != filename
                        or not labels or labels != reading):
                    skipped += 1
                    continue
                image_path = image_dir / filename
                if not image_path.is_file():
                    skipped += 1
                    continue
                task = AnnotationTask(quality, filename, labels, image_path)
                if all(_crop_path(output_dir, task, i).is_file() for i in range(len(labels))):
                    continue
                tasks.append(task)
    return tasks, skipped


def _page() -> bytes:
    return _HTML.encode("utf-8")


class _AnnotationServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address, handler, tasks, output_dir):
        super().__init__(address, handler)
        self.tasks = tasks
        self.output_dir = output_dir
        self.task_index = 0


def _handler_type():
    class Handler(BaseHTTPRequestHandler):
        server: _AnnotationServer

        def _send_json(self, status: int, data: dict) -> None:
            payload = json.dumps(data).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self) -> None:
            route = urlparse(self.path).path
            if route == "/":
                page = _page()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(page)))
                self.end_headers()
                self.wfile.write(page)
                return
            if route != "/api/next":
                self.send_error(404)
                return
            if self.server.task_index >= len(self.server.tasks):
                self._send_json(200, {"done": True})
                return
            task = self.server.tasks[self.server.task_index]
            image_data = base64.b64encode(task.image_path.read_bytes()).decode("ascii")
            self._send_json(200, {
                "done": False,
                "id": task.task_id,
                "filename": task.filename,
                "quality": task.quality,
                "labels": task.labels,
                "index": self.server.task_index + 1,
                "total": len(self.server.tasks),
                "image": image_data,
            })

        def do_POST(self) -> None:
            route = urlparse(self.path).path
            if route == "/api/skip":
                if self.server.task_index < len(self.server.tasks):
                    self.server.task_index += 1
                self._send_json(200, {"ok": True})
                return
            if route != "/api/annotate":
                self.send_error(404)
                return
            if self.server.task_index >= len(self.server.tasks):
                self._send_json(409, {"error": "There are no images left to annotate."})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                self._send_json(400, {"error": "Invalid request length."})
                return
            if length <= 0 or length > 100_000:
                self._send_json(400, {"error": "Request body must be between 1 and 100000 bytes."})
                return
            try:
                body = json.loads(self.rfile.read(length))
            except (json.JSONDecodeError, UnicodeDecodeError):
                self._send_json(400, {"error": "Request body must be valid JSON."})
                return
            task = self.server.tasks[self.server.task_index]
            if not isinstance(body, dict):
                self._send_json(400, {"error": "Request body must be a JSON object."})
                return
            boxes = body.get("boxes")
            if body.get("id") != task.task_id or not isinstance(boxes, list) or len(boxes) != len(task.labels):
                self._send_json(400, {"error": f"Draw exactly {len(task.labels)} digit boxes in left-to-right order."})
                return
            image = cv2.imread(str(task.image_path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                self._send_json(500, {"error": f"Could not read image: {task.image_path}"})
                return
            height, width = image.shape
            x_positions = []
            crops = []
            for box in boxes:
                if not isinstance(box, dict):
                    self._send_json(400, {"error": "Each digit box must contain x, y, width, and height."})
                    return
                try:
                    x, y, box_width, box_height = (
                        float(box[key]) for key in ("x", "y", "width", "height")
                    )
                except (KeyError, TypeError, ValueError):
                    self._send_json(400, {"error": "Each digit box must contain numeric x, y, width, and height."})
                    return
                if (not all(math.isfinite(value) for value in (x, y, box_width, box_height))
                        or x < 0 or y < 0 or box_width < 2 or box_height < 2
                        or x + box_width > width or y + box_height > height):
                    self._send_json(400, {"error": "A digit box is outside the image or has invalid dimensions."})
                    return
                x_positions.append(x)
                pad_x, pad_y = round(box_width * 0.08), round(box_height * 0.08)
                x0, y0 = max(0, round(x) - pad_x), max(0, round(y) - pad_y)
                x1 = min(width, round(x + box_width) + pad_x)
                y1 = min(height, round(y + box_height) + pad_y)
                crops.append(normalise_crop(image[y0:y1, x0:x1]))
            if any(left >= right for left, right in zip(x_positions, x_positions[1:])):
                self._send_json(400, {"error": "Draw digit boxes from left to right without reversing their order."})
                return
            for position, (label, crop) in enumerate(zip(task.labels, crops)):
                ok, encoded = cv2.imencode(".png", (crop * 255).astype("uint8"))
                if not ok:
                    self._send_json(500, {"error": "Could not encode a digit crop."})
                    return
                destination = _crop_path(self.server.output_dir, task, position)
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(encoded.tobytes())
            self.server.task_index += 1
            self._send_json(200, {"ok": True})

        def log_message(self, fmt, *args):
            return

    return Handler


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Annotate archived meter images for digit-model training.")
    parser.add_argument("--root", default="data/archive/Datasets", help="dataset directory containing CSV files and image folders")
    parser.add_argument("--out", default="data/annotated_digit_crops", help="output folder for labeled digit crops")
    parser.add_argument("--quality", nargs="+", choices=QUALITIES, default=list(QUALITIES))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true", help="do not open the annotation page automatically")
    args = parser.parse_args(argv)
    tasks, skipped = load_tasks(Path(args.root), Path(args.out), args.quality)
    if not tasks:
        print(f"No images need annotation. Skipped {skipped} rows with inconsistent labels or missing files.")
        return
    server = _AnnotationServer((args.host, args.port), _handler_type(), tasks, Path(args.out))
    url = f"http://{args.host}:{server.server_port}/"
    print(f"Annotating {len(tasks)} images; skipped {skipped} rows with inconsistent labels or missing files.")
    print(f"Open {url} and draw each labeled digit from left to right. Press Ctrl+C to stop; progress is saved.")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nAnnotation stopped. Saved digit crops can be resumed on the next run.")
    finally:
        server.server_close()


_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Meter digit annotation</title>
<style>
body{font:16px system-ui,sans-serif;max-width:1100px;margin:24px auto;padding:0 16px;color:#18212b;background:#f5f7fa}
header,.panel{background:white;border:1px solid #dce2e8;border-radius:10px;padding:16px;margin-bottom:16px}
h1{font-size:1.35rem;margin:0 0 8px}#status{color:#425466}#target{font-size:1.7rem;font-weight:700;letter-spacing:.18em;color:#075985}
#canvas-wrap{background:#e8edf2;padding:8px;overflow:auto;max-height:72vh}
canvas{display:block;max-width:100%;height:auto;max-height:70vh;cursor:crosshair;margin:auto}
button{font:inherit;padding:9px 14px;border:0;border-radius:6px;cursor:pointer;margin:8px 8px 0 0}
#save{background:#087e4b;color:white}#undo,#skip{background:#e7edf2;color:#18212b}
#message{min-height:1.5em;color:#a12b24}.hint{font-size:.92rem;color:#526170}
</style>
</head>
<body>
<header><h1>Label meter digits</h1><div id="status">Loading dataset…</div><div>Expected left-to-right digits: <span id="target"></span></div><div class="hint">Drag one box around each expected digit, in order. Include the full glyph but exclude neighboring digits and decimal marks.</div></header>
<section class="panel"><div id="canvas-wrap"><canvas id="image"></canvas></div>
<button id="undo">Undo last box</button><button id="skip">Skip image</button><button id="save">Save crops and continue</button><div id="message"></div></section>
<script>
const canvas=document.getElementById("image"),ctx=canvas.getContext("2d"),statusEl=document.getElementById("status"),targetEl=document.getElementById("target"),messageEl=document.getElementById("message");
let task=null,img=null,boxes=[],dragStart=null,dragEnd=null;
function draw(){if(!img)return;ctx.clearRect(0,0,canvas.width,canvas.height);ctx.drawImage(img,0,0);boxes.forEach((b,i)=>{ctx.strokeStyle="#00d084";ctx.lineWidth=Math.max(2,canvas.width/700);ctx.strokeRect(b.x,b.y,b.width,b.height);ctx.fillStyle="#00d084";ctx.font=`bold ${Math.max(16,canvas.width/35)}px sans-serif`;ctx.fillText(task.labels[i],b.x,b.y-5)});if(dragStart&&dragEnd){ctx.strokeStyle="#ff5c35";ctx.lineWidth=Math.max(2,canvas.width/700);ctx.strokeRect(dragStart.x,dragStart.y,dragEnd.x-dragStart.x,dragEnd.y-dragStart.y)}}
function point(e){const r=canvas.getBoundingClientRect();return{x:(e.clientX-r.left)*canvas.width/r.width,y:(e.clientY-r.top)*canvas.height/r.height}}
canvas.addEventListener("pointerdown",e=>{if(!task||boxes.length>=task.labels.length)return;canvas.setPointerCapture(e.pointerId);dragStart=point(e);dragEnd=dragStart;draw()});
canvas.addEventListener("pointermove",e=>{if(dragStart){dragEnd=point(e);draw()}});
canvas.addEventListener("pointerup",e=>{if(!dragStart)return;dragEnd=point(e);const x=Math.min(dragStart.x,dragEnd.x),y=Math.min(dragStart.y,dragEnd.y),width=Math.abs(dragEnd.x-dragStart.x),height=Math.abs(dragEnd.y-dragStart.y);if(width>=3&&height>=3)boxes.push({x,y,width,height});dragStart=null;dragEnd=null;draw();messageEl.textContent=`Selected ${boxes.length} of ${task.labels.length} digits.`});
async function loadTask(){messageEl.textContent="";boxes=[];dragStart=null;const r=await fetch("/api/next");const d=await r.json();task=d;if(d.done){statusEl.textContent="All selected images are annotated. You can close this tab.";targetEl.textContent="";ctx.clearRect(0,0,canvas.width,canvas.height);return}statusEl.textContent=`${d.index} of ${d.total} · ${d.quality} · ${d.filename}`;targetEl.textContent=d.labels;img=new Image();img.onload=()=>{canvas.width=img.naturalWidth;canvas.height=img.naturalHeight;draw()};img.src=`data:image/jpeg;base64,${d.image}`;messageEl.textContent="";}
document.getElementById("undo").onclick=()=>{boxes.pop();draw();messageEl.textContent=`Selected ${boxes.length} of ${task.labels.length} digits.`};
document.getElementById("skip").onclick=async()=>{await fetch("/api/skip",{method:"POST"});loadTask()};
document.getElementById("save").onclick=async()=>{if(!task)return;if(boxes.length!==task.labels.length){messageEl.textContent=`Draw ${task.labels.length} digit boxes first.`;return}const r=await fetch("/api/annotate",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({id:task.id,boxes})});const d=await r.json();if(!r.ok){messageEl.textContent=d.error||"Could not save annotation.";return}loadTask()};
loadTask().catch(e=>{statusEl.textContent="Could not load the annotation task.";messageEl.textContent=String(e)});
</script>
</body>
</html>"""


if __name__ == "__main__":
    main()
