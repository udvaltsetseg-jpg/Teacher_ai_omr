import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import google.generativeai as genai
from openai import OpenAI
import os
import json
from io import BytesIO
import zipfile
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import streamlit.components.v1 as components
from reportlab.lib import colors
from reportlab.lib.units import mm
import re
import cv2
import html as html_lib

try:
    from streamlit_cropper import st_cropper
except Exception:
    st_cropper = None

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

FONT_PATHS = [
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/Library/Fonts/Arial Unicode.ttf",
    "/Library/Fonts/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]

PDF_FONT = "Helvetica"

for path in FONT_PATHS:
    if os.path.exists(path):
        try:
            pdfmetrics.registerFont(TTFont("MongolianFont", path))
            PDF_FONT = "MongolianFont"
            break
        except Exception:
            pass

BLOOM_FULL = {
    "R": "Remember / Санах",
    "U": "Understand / Ойлгох",
    "A": "Apply / Хэрэглэх",
    "AN": "Analyze / Задлан шинжлэх",
    "E": "Evaluate / Үнэлэх",
    "C": "Create / Бүтээх",
}

BLOOM_MN = {
    "R": "Санах",
    "U": "Ойлгох",
    "A": "Хэрэглэх",
    "AN": "Задлан шинжлэх",
    "E": "Үнэлэх",
    "C": "Бүтээх",
}

st.set_page_config(
    page_title="Teacher AI OMR",
    layout="wide",
)

st.markdown("""
<style>
[data-testid="stSidebar"] { background-color: #EEF3FF; }
.hero {
    background: linear-gradient(135deg, #EEF3FF, #FFFFFF);
    padding: 28px;
    border-radius: 22px;
    border: 1px solid #E3E8FF;
    margin-bottom: 20px;
}
.hero h1 {
    margin: 0;
    font-size: 42px;
    font-weight: 850;
    color: #111827;
}
.hero p {
    margin-top: 8px;
    color: #6B7280;
    font-size: 16px;
}
div[data-testid="stMetric"] {
    background: white;
    padding: 18px;
    border-radius: 16px;
    border: 1px solid #E6E8F0;
    box-shadow: 0 4px 14px rgba(0,0,0,0.04);
}
.stButton button {
    border-radius: 12px;
    padding: 10px 18px;
    font-weight: 600;
}
.section-card {
    background:#ffffff;
    padding:24px;
    border-radius:22px;
    border:1px solid #E5E7EB;
    box-shadow:0 8px 24px rgba(0,0,0,0.04);
    margin-bottom:24px;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
    <h1>Teacher AI OMR</h1>
    <p>Phone Camera → AI Grading → Bloom Analytics → Parent Report → LXP</p>
</div>
""", unsafe_allow_html=True)

if "batch_results" not in st.session_state:
    st.session_state.batch_results = []
if "final_ai_recommendation" not in st.session_state:
    st.session_state.final_ai_recommendation = ""
if "gemini_result" not in st.session_state:
    st.session_state.gemini_result = ""
if "chatgpt_result" not in st.session_state:
    st.session_state.chatgpt_result = ""
if "parent_report_text" not in st.session_state:
    st.session_state.parent_report_text = ""
if "debug_image" not in st.session_state:
    st.session_state.debug_image = None
if "rotation_angle" not in st.session_state:
    st.session_state.rotation_angle = 0


def generate_answer_sheet_pdf(question_count):
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    c.setFont(PDF_FONT, 16)
    c.drawString(60, height - 50, "AI OMR Answer Sheet")

    c.setFont(PDF_FONT, 10)
    c.drawString(60, height - 75, "Student Code: ____________________")
    c.drawString(330, height - 75, "Name: ____________________")
    c.drawString(60, height - 95, "Instruction: Fill only one bubble for each question.")

    options = ["A", "B", "C", "D"]
    left_start_x = 60
    right_start_x = 320
    start_y = height - 130
    row_gap = 28
    option_gap = 45
    bubble_radius = 8
    rows_per_column = 24

    for i in range(question_count):
        index_on_page = i % (rows_per_column * 2)

        if i > 0 and index_on_page == 0:
            c.showPage()
            c.setFont(PDF_FONT, 16)
            c.drawString(60, height - 50, "AI OMR Answer Sheet")

        if index_on_page < rows_per_column:
            base_x = left_start_x
            row = index_on_page
        else:
            base_x = right_start_x
            row = index_on_page - rows_per_column

        y = start_y - row * row_gap
        c.drawString(base_x, y - 4, f"{i + 1}.")

        for j, opt in enumerate(options):
            x = base_x + 45 + j * option_gap
            c.circle(x, y, bubble_radius)
            c.drawString(x + 12, y - 4, opt)

    c.save()
    buffer.seek(0)
    return buffer


def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def find_answer_sheet(image_pil):
    img = np.array(image_pil.convert("RGB"))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    page_contour = None

    for cnt in contours[:15]:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4:
            page_contour = approx.reshape(4, 2)
            break

    if page_contour is None:
        return img

    rect = order_points(page_contour)
    target_w = 900
    target_h = 1273
    dst = np.array(
        [[0, 0], [target_w - 1, 0], [target_w - 1, target_h - 1], [0, target_h - 1]],
        dtype="float32",
    )
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(img, M, (target_w, target_h))
    return warped


def detect_omr_answers(image_pil, question_count):
    with st.sidebar.expander("OMR Debug гар тохиргоо", expanded=False):
        first_x = st.number_input("OMR: Эхний A bubble X", 50, 500, 178, 1, key="omr_first_x")
        first_y = st.number_input("OMR: Эхний мөр Y", 100, 700, 326, 1, key="omr_first_y")
        bubble_gap_x = st.number_input("OMR: A-B-C-D зай", 20, 120, 54, 1, key="omr_gap_x")
        question_gap_y = st.number_input("OMR: Асуулт хоорондын зай", 15, 80, 39, 1, key="omr_gap_y")
        bubble_radius = st.number_input("OMR: Bubble radius", 6, 35, 16, 1, key="omr_radius")
        threshold_ratio = st.number_input("OMR: Таних мэдрэмж", 0.05, 0.70, 0.46, 0.01, key="omr_threshold")
        show_debug = st.checkbox("OMR Debug View харуулах", value=True, key="omr_show_debug")

    img = np.array(image_pil.convert("RGB"))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    h0, w0 = img.shape[:2]

    if w0 > h0:
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

    target_h = 1400
    scale = target_h / img.shape[0]
    target_w = int(img.shape[1] * scale)
    img = cv2.resize(img, (target_w, target_h))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        12,
    )

    options = ["A", "B", "C", "D"]
    detected = []
    debug = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

    for q in range(int(question_count)):
        y = int(first_y + q * question_gap_y)
        scores = []

        for c_idx in range(4):
            x = int(first_x + c_idx * bubble_gap_x)
            mask = np.zeros(thresh.shape, dtype=np.uint8)
            cv2.circle(mask, (x, y), int(bubble_radius), 255, -1)
            roi = cv2.bitwise_and(thresh, thresh, mask=mask)
            filled = cv2.countNonZero(roi)
            total = np.pi * (int(bubble_radius) ** 2)
            ratio = filled / total
            scores.append(ratio)
            cv2.circle(debug, (x, y), int(bubble_radius), (0, 255, 0), 2)
            cv2.putText(debug, f"{ratio:.2f}", (x - 16, y + 34), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 255), 1)

        selected = [i for i, s in enumerate(scores) if s >= threshold_ratio]

        if len(selected) == 1:
            idx = selected[0]
            detected.append(options[idx])
            x = int(first_x + idx * bubble_gap_x)
            cv2.circle(debug, (x, y), int(bubble_radius) + 5, (255, 0, 0), 3)
        elif len(selected) > 1:
            detected.append("MULTI")
        else:
            max_score = max(scores)
            if max_score >= threshold_ratio * 0.75:
                idx = int(np.argmax(scores))
                detected.append(options[idx])
            else:
                detected.append("")

    if show_debug:
        st.session_state.debug_image = cv2.cvtColor(debug, cv2.COLOR_BGR2RGB)
    else:
        st.session_state.debug_image = None

    while len(detected) < question_count:
        detected.append("")

    return detected[:question_count]


def clean_markdown(text):
    text = text.replace("###", "")
    text = text.replace("**", "")
    text = text.replace("*", "")
    return text.strip()


def draw_wrapped_line(c, text, x, y, max_width, font_name, font_size, line_height):
    words = text.split(" ")
    line = ""

    for word in words:
        test_line = line + word + " "
        if c.stringWidth(test_line, font_name, font_size) <= max_width:
            line = test_line
        else:
            c.drawString(x, y, line.strip())
            y -= line_height
            line = word + " "
            if y < 25 * mm:
                c.showPage()
                y = A4[1] - 25 * mm
                c.setFont(font_name, font_size)

    if line:
        c.drawString(x, y, line.strip())
        y -= line_height

    return y


def generate_parent_report_pdf(report_text):
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4
    margin_left = 25 * mm
    margin_right = 25 * mm
    margin_top = 22 * mm
    max_width = width - margin_left - margin_right
    y = height - margin_top
    report_text = clean_markdown(report_text)

    c.setFillColor(colors.HexColor("#1F2937"))
    c.setFont(PDF_FONT, 18)
    c.drawString(margin_left, y, "Teacher AI Parent Feedback Report")
    y -= 8 * mm
    c.setStrokeColor(colors.HexColor("#CBD5E1"))
    c.line(margin_left, y, width - margin_right, y)
    y -= 10 * mm
    c.setFillColor(colors.black)
    c.setFont(PDF_FONT, 10)

    for raw_line in report_text.split("\n"):
        line = raw_line.strip()
        if line == "":
            y -= 4 * mm
            continue
        if y < 25 * mm:
            c.showPage()
            y = height - margin_top
            c.setFont(PDF_FONT, 10)
        if "====" in line:
            y -= 3 * mm
            c.setStrokeColor(colors.HexColor("#E5E7EB"))
            c.line(margin_left, y, width - margin_right, y)
            y -= 6 * mm
            continue
        if line.startswith("Bloom Analysis") or line.startswith("AI Recommendation"):
            y -= 3 * mm
            c.setFillColor(colors.HexColor("#111827"))
            c.setFont(PDF_FONT, 13)
            y = draw_wrapped_line(c, line, margin_left, y, max_width, PDF_FONT, 13, 16)
            y -= 2 * mm
            c.setFont(PDF_FONT, 10)
            c.setFillColor(colors.black)
            continue
        if line.startswith("Сурагчийн код") or line.startswith("Сурагчийн нэр") or line.startswith("Оноо") or line.startswith("Гүйцэтгэл"):
            c.setFillColor(colors.HexColor("#374151"))
            c.setFont(PDF_FONT, 10)
            y = draw_wrapped_line(c, line, margin_left, y, max_width, PDF_FONT, 10, 14)
            c.setFillColor(colors.black)
            continue
        if line.startswith("-"):
            y = draw_wrapped_line(c, "• " + line[1:].strip(), margin_left + 6 * mm, y, max_width - 6 * mm, PDF_FONT, 10, 14)
            continue
        if re.match(r"^\d+\.", line):
            c.setFillColor(colors.HexColor("#111827"))
            c.setFont(PDF_FONT, 11)
            y = draw_wrapped_line(c, line, margin_left, y, max_width, PDF_FONT, 11, 15)
            c.setFont(PDF_FONT, 10)
            c.setFillColor(colors.black)
            continue
        y = draw_wrapped_line(c, line, margin_left, y, max_width, PDF_FONT, 10, 14)

    c.setFont(PDF_FONT, 8)
    c.setFillColor(colors.HexColor("#6B7280"))
    c.drawString(margin_left, 12 * mm, "Generated by Teacher AI OMR")
    c.save()
    buffer.seek(0)
    return buffer


def make_prompt(bloom_stats, percent, student_name):
    return f"""
Сурагчийн нэр: {student_name}

Bloom анализ:

{json.dumps(bloom_stats, ensure_ascii=False)}

Нийт хувь: {percent}%

Монгол хэлээр дараах бүтэцтэй зөвлөмж гарга:

1. Давуу тал
2. Сул тал
3. Сайжруулах арга
4. Багшид өгөх зөвлөгөө
5. Сурагчид хэлэх эерэг feedback
"""


def make_parent_report_text(student_code, student_name, score, total, percent, bloom_stats, ai_text):
    bloom_lines = "\n".join([f"- {BLOOM_FULL.get(k, k)}: {v}%" for k, v in bloom_stats.items()])
    return f"""
Сурагчийн код: {student_code}

Сурагчийн нэр: {student_name}

Оноо: {score}/{total}

Гүйцэтгэл: {percent}%

========================================

Bloom Analysis / Bloom түвшний шинжилгээ

========================================

{bloom_lines}

========================================

AI Recommendation / AI зөвлөмж

========================================

{ai_text if ai_text else "AI зөвлөмж үүсээгүй байна."}
"""







def create_lxp_extension_zip():
    buffer = BytesIO()

    manifest_json = {
        "manifest_version": 3,
        "name": "Teacher AI LXP Connector",
        "version": "1.5",
        "description": "Teacher AI OMR Batch JSON-г clipboard/storage-оос уншаад LXP дээр AUTO FILL хийх connector.",
        "permissions": ["tabs", "scripting", "storage", "activeTab", "clipboardRead"],
        "host_permissions": [
            "http://localhost:8501/*",
            "https://*.streamlit.app/*",
            "https://share.streamlit.io/*",
            "https://lxp.eschool.mn/*"
        ],
        "background": {
            "service_worker": "background.js"
        },
        "content_scripts": [
            {
                "matches": [
                    "http://localhost:8501/*",
                    "https://*.streamlit.app/*",
                    "https://share.streamlit.io/*"
                ],
                "js": ["streamlit_content.js"],
                "all_frames": True,
                "run_at": "document_idle"
            },
            {
                "matches": ["https://lxp.eschool.mn/*"],
                "js": ["lxp_content.js"],
                "run_at": "document_idle"
            }
        ],
        "action": {
            "default_popup": "popup.html",
            "default_title": " LXP Connector"
        }
    }

    popup_html = """<!DOCTYPE html>
<html lang="mn">
<head>
  <meta charset="UTF-8" />
  <title> LXP Connector</title>
  <link rel="stylesheet" href="style.css" />
</head>
<body>
  <div class="card">
    <h2> LXP Connector</h2>
    <p class="hint">
      Streamlit дээр <b>SEND ALL TO LXP</b> дарсан бол энэ товчоор LXP дээр шууд бөглөнө.
    </p>

    <button id="fillLatestBtn">AUTO FILL LXP</button>

    <div id="status"></div>
  </div>
  <script src="popup.js"></script>
</body>
</html>
"""

    style_css = """body{width:340px;margin:0;font-family:Arial,sans-serif;background:#EEF3FF;color:#111827}.card{padding:16px}h2{margin:0 0 8px 0;font-size:20px}.hint{font-size:12px;color:#6B7280;line-height:1.55}button{width:100%;border:none;border-radius:12px;padding:14px;font-weight:800;cursor:pointer;background:#16A085;color:white;margin-top:10px;font-size:15px}#status{margin-top:12px;font-size:12px;border-radius:10px;padding:10px;display:none;line-height:1.45}"""

    popup_js = """const fillLatestBtn = document.getElementById("fillLatestBtn");
const statusBox = document.getElementById("status");

function showStatus(message, isError=false) {
  statusBox.style.display = "block";
  statusBox.style.background = isError ? "#FEE2E2" : "#D1FAE5";
  statusBox.style.color = isError ? "#991B1B" : "#065F46";
  statusBox.textContent = message;
}

async function getPayloadFromClipboard() {
  try {
    const text = await navigator.clipboard.readText();
    const payload = JSON.parse(text);
    if (Array.isArray(payload)) return payload;
  } catch (e) {
    console.warn("Clipboard read/parse failed:", e);
  }
  return [];
}

fillLatestBtn.addEventListener("click", async () => {
  chrome.storage.local.get(["latestPayload"], async (result) => {
    let payload = result.latestPayload || [];

    if (!Array.isArray(payload) || payload.length === 0) {
      payload = await getPayloadFromClipboard();
    }

    if (!Array.isArray(payload) || payload.length === 0) {
      showStatus("Batch JSON олдсонгүй. Streamlit дээр SEND ALL TO LXP дарсны дараа дахин AUTO FILL LXP дарна уу.", true);
      return;
    }

    chrome.storage.local.set({ latestPayload: payload });

    chrome.runtime.sendMessage({ action: "FILL_LXP", payload }, (response) => {
      if (chrome.runtime.lastError) {
        showStatus(chrome.runtime.lastError.message, true);
        return;
      }

      if (response && response.ok) {
        showStatus(`Амжилттай: ${response.filled} мөр. Алгассан: ${response.skipped} мөр.`);
      } else {
        showStatus(response?.message || "Алдаа гарлаа.", true);
      }
    });
  });
});
"""

    streamlit_content_js = """console.log("✅ Streamlit content loaded v1.5");

window.addEventListener("message", (event) => {
  if (!event.data) return;

  const validSource =
    event.data.source === "TEACHER_AI" ||
    event.data.source === "TEACHER_AI";

  if (!validSource) return;

  if (event.data.action === "SAVE_LXP_PAYLOAD") {
    chrome.runtime.sendMessage({
      action: "SAVE_LXP_PAYLOAD",
      payload: event.data.payload
    });
    return;
  }

  if (event.data.action === "SEND_TO_LXP") {
    chrome.runtime.sendMessage({
      action: "SAVE_AND_FILL_LXP",
      payload: event.data.payload
    });
  }
});
"""

    background_js = """console.log("✅ Teacher AI LXP Connector background v1.5 loaded");

function fillLxp(payload, sendResponse) {
  chrome.tabs.query({ url: "https://lxp.eschool.mn/*" }, (tabs) => {
    if (!tabs || tabs.length === 0) {
      const result = {
        ok: false,
        message: "LXP tab олдсонгүй. Эхлээд LXP дүн оруулах page нээнэ үү.",
        filled: 0,
        skipped: Array.isArray(payload) ? payload.length : 0,
        errors: [{ code: "-", score: "", message: "LXP tab олдсонгүй" }]
      };
      if (sendResponse) sendResponse(result);
      return;
    }

    const lxpTab = tabs[0];

    chrome.tabs.sendMessage(
      lxpTab.id,
      {
        action: "FILL_LXP_FROM_STREAMLIT",
        payload: payload
      },
      (response) => {
        if (chrome.runtime.lastError) {
          const result = {
            ok: false,
            message: "LXP page refresh хийгээд дахин оролдоно уу. " + chrome.runtime.lastError.message,
            filled: 0,
            skipped: Array.isArray(payload) ? payload.length : 0,
            errors: [{ code: "-", score: "", message: chrome.runtime.lastError.message }]
          };
          if (sendResponse) sendResponse(result);
          return;
        }

        const result = {
          ok: response?.ok ?? false,
          message: response?.message || "",
          filled: response?.filled || 0,
          skipped: response?.skipped || 0,
          errors: response?.errors || []
        };

        if (sendResponse) sendResponse(result);
      }
    );

    chrome.tabs.update(lxpTab.id, { active: true });
  });
}

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (!message) return;

  if (message.action === "SAVE_LXP_PAYLOAD") {
    const payload = message.payload || [];
    chrome.storage.local.set({ latestPayload: payload }, () => {
      if (sendResponse) sendResponse({ ok: true, count: payload.length });
    });
    return true;
  }

  if (message.action === "SAVE_AND_FILL_LXP" || message.action === "FILL_LXP") {
    const payload = message.payload || [];
    chrome.storage.local.set({ latestPayload: payload }, () => {
      fillLxp(payload, sendResponse);
    });
    return true;
  }
});
"""

    lxp_content_js = """console.log("✅ LXP content loaded v1.5");

function normalizeText(value) {
  return String(value || "").trim().toLowerCase();
}

function normalizeCode(value) {
  return String(value || "").trim().toUpperCase();
}

function validateScore(score) {
  if (score === null || score === undefined || String(score).trim() === "") {
    return { ok: false, message: "хоосон дүн" };
  }

  const raw = String(score).trim().replace(",", ".");

  if (/[A-Za-zА-Яа-яӨөҮүЁё]/.test(raw)) {
    return { ok: false, message: "үсгэн дүн байна" };
  }

  const num = Number(raw);

  if (Number.isNaN(num)) return { ok: false, message: "тоон дүн биш" };
  if (num < 0) return { ok: false, message: "0-оос бага дүн" };
  if (num > 100) return { ok: false, message: "100-аас их дүн" };

  return { ok: true, value: num };
}

function setInputValue(input, value) {
  const setter = Object.getOwnPropertyDescriptor(
    window.HTMLInputElement.prototype,
    "value"
  )?.set;

  if (setter) setter.call(input, value);
  else input.value = value;

  input.dispatchEvent(new Event("input", { bubbles: true }));
  input.dispatchEvent(new Event("change", { bubbles: true }));
  input.dispatchEvent(new Event("blur", { bubbles: true }));
}

function markInput(input, ok, title) {
  input.style.outline = ok ? "3px solid #22C55E" : "3px solid #EF4444";
  input.style.backgroundColor = ok ? "#ECFDF5" : "#FEF2F2";
  input.title = title || "";
}

function findStudentRow(student) {
  const code = normalizeCode(student.code || student.StudentCode || student.student_code);
  const name = normalizeText(student.name || student.StudentName || student.student_name);

  const rows = Array.from(document.querySelectorAll("tr"));

  for (const row of rows) {
    const text = normalizeText(row.innerText);
    const upperText = normalizeCode(row.innerText);

    if (code && upperText.includes(code)) return row;
    if (name && name !== "nan" && text.includes(name)) return row;
  }

  return null;
}

function findScoreInputInRow(row) {
  const inputs = Array.from(row.querySelectorAll("input"));

  const editable = inputs.filter((input) => {
    if (input.disabled || input.readOnly) return false;
    const type = (input.getAttribute("type") || "text").toLowerCase();
    return ["number", "text", "tel"].includes(type);
  });

  if (editable.length === 0) return null;

  return editable[editable.length - 1];
}

function fillLXP(payload) {
  let filled = 0;
  let skipped = 0;
  const errors = [];

  if (!Array.isArray(payload)) {
    return {
      ok: false,
      filled: 0,
      skipped: 0,
      errors: [{ code: "-", message: "payload array биш байна" }],
      message: "JSON array биш байна."
    };
  }

  payload.forEach((student) => {
    const code = student.code || student.StudentCode || student.student_code || "";
    const name = student.name || student.StudentName || student.student_name || "";
    const score = student.score ?? student.Score ?? student.grade ?? student.Grade;

    const scoreCheck = validateScore(score);

    if (!scoreCheck.ok) {
      skipped++;
      errors.push({ code: code || name || "unknown", score, message: scoreCheck.message });
      return;
    }

    const row = findStudentRow(student);

    if (!row) {
      skipped++;
      errors.push({ code: code || name || "unknown", score, message: "LXP жагсаалтаас сурагч олдсонгүй" });
      return;
    }

    const input = findScoreInputInRow(row);

    if (!input) {
      skipped++;
      errors.push({ code: code || name || "unknown", score, message: "оноо оруулах input олдсонгүй" });
      return;
    }

    setInputValue(input, String(scoreCheck.value));
    markInput(input, true, " Teacher AI бөглөсөн");
    filled++;
  });

  return {
    ok: true,
    filled,
    skipped,
    errors,
    message: `Амжилттай ${filled} мөр бөглөгдлөө. ${skipped} мөр алгаслаа.`
  };
}

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (!message) return;

  if (message.action === "FILL_LXP_FROM_STREAMLIT") {
    const result = fillLXP(message.payload || []);
    sendResponse(result);
  }

  return true;
});
"""

    readme_md = """# Teacher AI LXP Connector v1.5

## Workflow
1. Streamlit дээр Batch List үүсгэнэ.
2. SEND ALL TO LXP дарна. Энэ үед JSON clipboard руу copy болно.
3. LXP tab дээр extension icon → AUTO FILL LXP дарна.
4. Extension clipboard/storage-оос JSON уншаад LXP дээр бөглөнө.

Paste хийх шаардлагагүй.
"""

    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("lxp_clipboard_connector/manifest.json", json.dumps(manifest_json, ensure_ascii=False, indent=2))
        zf.writestr("lxp_clipboard_connector/popup.html", popup_html)
        zf.writestr("lxp_clipboard_connector/style.css", style_css)
        zf.writestr("lxp_clipboard_connector/popup.js", popup_js)
        zf.writestr("lxp_clipboard_connector/streamlit_content.js", streamlit_content_js)
        zf.writestr("lxp_clipboard_connector/background.js", background_js)
        zf.writestr("lxp_clipboard_connector/lxp_content.js", lxp_content_js)
        zf.writestr("lxp_clipboard_connector/README.md", readme_md)

    buffer.seek(0)
    return buffer


with st.sidebar:
    st.header("Answer Sheet")
    q_count = st.number_input("Асуултын тоо", value=20, min_value=1, max_value=200)
    pdf_buffer = generate_answer_sheet_pdf(q_count)
    st.download_button(
        label="PDF татах",
        data=pdf_buffer,
        file_name=f"OMR_{q_count}Q.pdf",
        mime="application/pdf",
    )

    preview_width = st.slider(
        "Preview зурагны өргөн",
        min_value=500,
        max_value=1400,
        value=1050,
        step=50,
    )

    st.divider()
    st.subheader("LXP Connector Extension")

    st.caption(
        "LXP Connector extension татаж суулгаснаар SEND ALL TO LXP товчоор дүнг LXP рүү шууд автоматаар бөглөнө."
    )

    lxp_extension_zip = create_lxp_extension_zip()

    st.download_button(
        label="⬇️ LXP Connector татах",
        data=lxp_extension_zip.getvalue(),
        file_name="lxp_clipboard_connector.zip",
        mime="application/zip",
        help="ZIP-г татаж аваад задлаад Chrome → Extensions → Load unpacked ашиглан суулгана."
    )

    with st.expander("Суулгах богино заавар", expanded=False):
        st.markdown("""
1. ZIP файлыг татаж аваад задлана.  
2. Chrome дээр `chrome://extensions` нээнэ.  
3. `Developer mode` асаана.  
4. `Load unpacked` дарна.  
5. Задалсан `lxp_clipboard_connector` folder-ийг сонгоно.  
6. Streamlit дээр SEND ALL TO LXP дараад, LXP дээр extension icon → AUTO FILL LXP дарна.  
""")


# ============================================
# APP MODE
# ============================================

app_mode = st.radio(
    "Ажиллах горим",
    [
        "📁 Бэлэн Excel → LXP",
        "📷 OMR шалгалт засах → LXP",
    ],
    horizontal=True,
)

answer_excel = None
answer_img = None
student_code = ""
student_name = ""

if app_mode == "📷 OMR шалгалт засах → LXP":
    col1, col2 = st.columns(2)

    with col1:
        answer_excel = st.file_uploader("Зөв хариултын Excel", type=["xlsx"])

    with col2:
        answer_img = st.file_uploader("Сурагчийн зураг", type=["jpg", "jpeg", "png"])

    student_code = st.text_input("Сурагчийн код", value="NEST25080001")
    student_name = st.text_input("Сурагчийн нэр", value="Сурагч")

# ============================================
# DIRECT EXCEL → LXP MODE
# ============================================

if app_mode == "📁 Бэлэн Excel → LXP":

    st.subheader("📁 Бэлэн Excel → LXP")

    with st.container(border=True):
        lxp_excel = st.file_uploader(
            "LXP-д шууд оруулах Excel файл",
            type=["xlsx"],
            key="direct_lxp_excel",
        )

        if lxp_excel is not None:
            lxp_df_original = pd.read_excel(lxp_excel)

            st.success("Excel амжилттай уншигдлаа")

            st.dataframe(
                lxp_df_original.head(20),
                use_container_width=True,
            )

            cols = list(lxp_df_original.columns)

            col_map1, col_map2, col_map3 = st.columns(3)

            with col_map1:
                code_col = st.selectbox(
                    "Student code column",
                    cols,
                    key="direct_lxp_code_col",
                )

            with col_map2:
                name_col = st.selectbox(
                    "Student name column",
                    cols,
                    key="direct_lxp_name_col",
                )

            with col_map3:
                score_col = st.selectbox(
                    "Score column",
                    cols,
                    key="direct_lxp_score_col",
                )

            lxp_ready_df = pd.DataFrame({
                "StudentCode": lxp_df_original[code_col],
                "StudentName": lxp_df_original[name_col],
                "Score": lxp_df_original[score_col],
            })

            st.markdown("### ✅ LXP Ready Preview")

            st.dataframe(
                lxp_ready_df,
                use_container_width=True,
            )

            add_col, clear_col = st.columns([1, 1])

            with add_col:
                if st.button("➕ Excel-с Batch List рүү нэмэх", key="direct_add_excel_to_batch"):
                    added_count = 0
                    skipped_count = 0

                    for _, row in lxp_ready_df.iterrows():
                        code = str(row["StudentCode"]).strip()
                        name = str(row["StudentName"]).strip()

                        try:
                            score = float(row["Score"])
                        except Exception:
                            score = 0

                        if code == "" or code.lower() == "nan":
                            skipped_count += 1
                            continue

                        exists = any(
                            str(x.get("code", "")).strip() == code
                            for x in st.session_state.batch_results
                        )

                        if exists:
                            skipped_count += 1
                        else:
                            st.session_state.batch_results.append({
                                "code": code,
                                "name": name,
                                "score": score,
                            })
                            added_count += 1

                    st.success(
                        f"{added_count} сурагч Batch List-д нэмэгдлээ. "
                        f"{skipped_count} мөр алгасагдлаа."
                    )
                    st.rerun()

            with clear_col:
                if st.button("🧹 Batch List цэвэрлэх", key="direct_clear_batch_top"):
                    st.session_state.batch_results = []
                    st.rerun()

            output = BytesIO()

            with pd.ExcelWriter(output, engine="openpyxl") as writer:
                lxp_ready_df.to_excel(
                    writer,
                    index=False,
                )

            st.download_button(
                "⬇️ Download LXP Ready Excel",
                data=output.getvalue(),
                file_name="LXP_READY.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

        else:
            st.info("LXP-д оруулах Excel файлаа upload хийнэ үү.")

    st.subheader("📋 Batch List / LXP Fill")

    if len(st.session_state.batch_results) > 0:
        batch_df = pd.DataFrame(st.session_state.batch_results)
        st.dataframe(batch_df, use_container_width=True, hide_index=True)

        payload_json = json.dumps(st.session_state.batch_results, ensure_ascii=False)
        components.html(
            f"""
<script>
const LXP_PAYLOAD = {payload_json};

function postToExtension(action) {{
    const msg = {{
        source: "TEACHER_AI",
        action: action,
        payload: LXP_PAYLOAD
    }};

    try {{ window.postMessage(msg, "*"); }} catch(e) {{}}
    try {{ window.parent.postMessage(msg, "*"); }} catch(e) {{}}
    try {{ window.top.postMessage(msg, "*"); }} catch(e) {{}}
}}

postToExtension("SAVE_LXP_PAYLOAD");
setTimeout(() => postToExtension("SAVE_LXP_PAYLOAD"), 500);
setTimeout(() => postToExtension("SAVE_LXP_PAYLOAD"), 1500);
setTimeout(() => postToExtension("SAVE_LXP_PAYLOAD"), 3000);
</script>
<button onclick="sendAllToLXP()"
    style="
        background:#16a085;
        color:white;
        border:none;
        padding:14px 22px;
        border-radius:12px;
        font-weight:bold;
        cursor:pointer;
        font-size:16px;
        width:100%;
    ">
    SEND ALL TO LXP
</button>

<script>
async function sendAllToLXP(){{
    const textPayload = JSON.stringify(LXP_PAYLOAD);

    try {{
        await navigator.clipboard.writeText(textPayload);
    }} catch(e) {{
        console.log("Clipboard copy failed", e);
    }}

    postToExtension("SEND_TO_LXP");
    alert("Batch list clipboard-д хадгалагдлаа. Одоо LXP tab дээр extension → AUTO FILL LXP дарна.");
}}
</script>
""",
            height=80,
        )

        if st.button("CLEAR BATCH", key="direct_clear_batch_bottom"):
            st.session_state.batch_results = []
            st.rerun()
    else:
        st.info("Batch list хоосон байна.")

if app_mode == "📷 OMR шалгалт засах → LXP" and answer_excel and answer_img:
    df_answer = pd.read_excel(answer_excel)

    if "Answer" not in df_answer.columns:
        st.error("Excel дээр 'Answer' багана байх ёстой.")
        st.stop()

    answers = df_answer["Answer"].astype(str).str.upper().tolist()

    if "Bloom" in df_answer.columns:
        bloom_list = df_answer["Bloom"].astype(str).str.upper().tolist()
    else:
        bloom_list = ["R", "U", "A", "AN", "E", "C"] * 100

    image = Image.open(answer_img).convert("RGB")

    # =========================================
    # ROTATION CONTROLS
    # Энд зураг эргүүлсний дараа OMR танилтыг дахин ажиллуулна
    # =========================================

    with st.container(border=True):
        st.subheader("Зураг эргүүлэх тохиргоо")

        rotate_col1, rotate_col2, rotate_col3, rotate_col4 = st.columns(4)

        with rotate_col1:
            if st.button("↩ 90° Left", key="rotate_left"):
                st.session_state.rotation_angle += 90
                st.rerun()

        with rotate_col2:
            if st.button("↪ 90° Right", key="rotate_right"):
                st.session_state.rotation_angle -= 90
                st.rerun()

        with rotate_col3:
            if st.button("🔄 180°", key="rotate_180"):
                st.session_state.rotation_angle += 180
                st.rerun()

        with rotate_col4:
            if st.button("Reset", key="rotate_reset"):
                st.session_state.rotation_angle = 0
                st.rerun()

        st.caption(f"Одоогийн эргэлт: {st.session_state.rotation_angle % 360}°")

    # Эргүүлсэн зураг
    rotated_image = image.rotate(
        st.session_state.rotation_angle,
        expand=True
    )

    # =========================================
    # CROP CONTROLS
    # Зураг upload хийсэн хэсэг дээр crop хийж, OMR-г crop/rotate хийсэн зураг дээр уншуулна
    # =========================================

    with st.container(border=True):
        st.subheader("✂️ Зураг crop хийх")

        crop_on = st.checkbox(
            "Crop хийж зөвхөн answer sheet хэсгийг сонгох",
            value=False,
            key="crop_on"
        )

        if crop_on:
            if st_cropper is None:
                st.error(
                    "streamlit-cropper суулгаагүй байна. Terminal дээр: "
                    "python3 -m pip install streamlit-cropper"
                )
                display_image = rotated_image
            else:
                st.caption(
                    "Ногоон хүрээг answer sheet-ийн гадна хүрээтэй тааруулна. "
                    "Crop хийсний дараа OMR танилт энэ crop зураг дээр ажиллана."
                )

                display_image = st_cropper(
                    rotated_image,
                    realtime_update=True,
                    box_color="#22C55E",
                    aspect_ratio=None,
                    return_type="image",
                    key="answer_sheet_cropper"
                )
        else:
            display_image = rotated_image

    # OMR танилтыг ЗААВАЛ rotate/crop хийсэн зураг дээр хийнэ
    ai_answers = detect_omr_answers(
        display_image,
        min(q_count, len(answers))
    )

    # =========================================
    # IMAGE + DEBUG PREVIEW
    # =========================================

    preview_col1, preview_col2 = st.columns(2)

    with preview_col1:
        with st.container(border=True):
            st.subheader("1. Uploaded Image")
            st.caption("Сурагчийн answer sheet preview")

            st.image(
                display_image,
                width=preview_width
            )

    with preview_col2:
        with st.container(border=True):
            st.subheader("OMR Debug Preview")
            st.caption("Ногоон bubble дээр яг таарч байвал зөв уншиж байна.")

            if st.session_state.debug_image is not None:
                debug_preview = Image.fromarray(
                    st.session_state.debug_image
                )

                st.image(
                    debug_preview,
                    width=preview_width
                )

            else:
                st.info("Debug image байхгүй байна.")

    results = []
    for i in range(min(q_count, len(answers))):
        is_correct = ai_answers[i] == answers[i]
        results.append(
            {
                "Q": i + 1,
                "AI_Answer": ai_answers[i],
                "Correct_Answer": answers[i],
                "Bloom": bloom_list[i],
                "Status": "✅ Зөв" if is_correct else "❌ Буруу",
            }
        )

    result_df = pd.DataFrame(results)

    row2_col1, row2_col2 = st.columns([1, 1])

    with row2_col1:
        with st.container(border=True):
            st.subheader("2. AI Grading Result")
            st.dataframe(result_df, use_container_width=True, height=760, hide_index=True)

    with row2_col2:
        with st.container(border=True):
            st.subheader("3. Teacher Review")
            review_mode = st.radio(
                "Review хийх хэлбэр",
                ["Бүх хариулт", "Зөвхөн буруу хариулт"],
                horizontal=True,
                key="review_mode",
            )

            final_answers = ai_answers[: len(results)]
            wrong_count = 0

            with st.container(height=700):
                for i in range(len(results)):
                    ai = ai_answers[i]
                    correct = answers[i]
                    is_wrong = ai != correct

                    if is_wrong:
                        wrong_count += 1

                    if review_mode == "Зөвхөн буруу хариулт" and not is_wrong:
                        continue

                    default_index = 0
                    if ai in ["A", "B", "C", "D"]:
                        default_index = ["", "A", "B", "C", "D"].index(ai)

                    selected = st.selectbox(
                        f"Q{i + 1} | AI: {ai}",
                        ["", "A", "B", "C", "D"],
                        index=default_index,
                        key=f"review_filter_{i}",
                    )
                    final_answers[i] = selected if selected != "" else ai

            if review_mode == "Зөвхөн буруу хариулт":
                st.info(f"Буруу эсвэл шалгах шаардлагатай: {wrong_count} асуулт")

    final_scores = []
    total_score = 0

    for i in range(len(results)):
        correct = final_answers[i] == answers[i]
        score = 1 if correct else 0
        total_score += score
        final_scores.append(score)

    percent = round(total_score / len(results) * 100, 1)

    with st.container(border=True):
        st.subheader("4. Final Score")
        correct_count = total_score
        wrong_count = len(results) - total_score
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Оноо", f"{total_score}/{len(results)}")
        m2.metric("Хувь", f"{percent}%")
        m3.metric("Зөв", correct_count)
        m4.metric("Буруу", wrong_count)

    bloom_levels = ["R", "U", "A", "AN", "E", "C"]
    bloom_stats = {}

    for level in bloom_levels:
        idxs = [idx for idx, val in enumerate(bloom_list[: len(results)]) if val == level]
        if len(idxs) == 0:
            bloom_stats[level] = 0
        else:
            val = np.mean([final_scores[x] for x in idxs]) * 100
            bloom_stats[level] = round(val, 1)

    chart_df = pd.DataFrame({"Bloom": list(bloom_stats.keys()), "Score": list(bloom_stats.values())})

    with st.container(border=True):
        st.subheader("5. Bloom Analytics")
        st.bar_chart(chart_df.set_index("Bloom"))

    with st.container(border=True):
        st.subheader("6. Rule-based зөвлөмж")
    weak_levels = [(level, score) for level, score in bloom_stats.items() if score < 60]

    if weak_levels:
        html = """
<style>
body{font-family:Arial,sans-serif;padding:14px;}
.rule-card{background:#ffffff;padding:28px;border-radius:20px;border:1px solid #E5E7EB;}
.rule-title{font-size:26px;font-weight:800;color:#111827;margin-bottom:16px;}
.rule-warning{background:#FEF3C7;color:#92400E;padding:14px;border-radius:12px;margin-bottom:18px;font-weight:700;}
.rule-badge{display:inline-block;background:#EEF2FF;color:#4338CA;padding:8px 14px;border-radius:10px;margin:5px;font-weight:700;font-size:15px;}
.rule-info{background:#EFF6FF;color:#1D4ED8;padding:18px;border-radius:14px;margin-top:18px;line-height:1.9;font-size:16px;max-height:120px;overflow-y:auto;}
.rule-info::-webkit-scrollbar{width:8px;}
.rule-info::-webkit-scrollbar-thumb{background:#CBD5E1;border-radius:10px;}
</style>
<div class="rule-card">
<div class="rule-title">⚠️ Сайжруулах шаардлагатай Bloom түвшин</div>
<div class="rule-warning">Дараах чадварууд сул үзүүлэлттэй байна</div>
"""
        for level, score in weak_levels:
            level_name = BLOOM_MN.get(level, level)
            html += f'<span class="rule-badge">{level_name} — {score}%</span>'

        html += """
<div class="rule-info">
<b>📘 Дэмжих арга</b><br><br>
Ойлголтыг дахин тайлбарлах<br>
Богино давтлага ажиллуулах<br>
Алдаатай жишээ засуулах<br>
Ижил төрлийн нэмэлт дасгал өгөх<br>
Pair work болон interactive activity ашиглах<br>
Interactive quiz ашиглан ахиц хянах<br>
Bloom түвшин ахиулах шаталсан даалгавар өгөх
</div>
</div>
"""
        components.html(html, height=320, scrolling=True)
    else:
        st.success("✅ Бүх Bloom түвшин хэвийн байна.")

    bloom_summary = ""
    for k, v in bloom_stats.items():
        full_name = BLOOM_FULL.get(k, k)
        bloom_summary += f"{full_name}: {v}%\n"

    st.subheader("7. Gemini + ChatGPT AI Recommendation")

    if st.button("AI зөвлөмж үүсгэх"):
        prompt = make_prompt(bloom_stats, percent, student_name)

        st.markdown("""
<style>
.ai-card{
    background:#ECFDF5;
    padding:24px;
    border-radius:18px;
    color:#166534;
    line-height:1.9;
    font-size:16px;
    border:1px solid #BBF7D0;
    height:620px;
    overflow-y:auto;
}
.ai-card::-webkit-scrollbar{width:8px;}
.ai-card::-webkit-scrollbar-thumb{background:#CBD5E1;border-radius:10px;}
</style>
""", unsafe_allow_html=True)

        col_gemini, col_chatgpt = st.columns(2)

        with col_gemini:
            st.markdown("# Gemini AI")
            try:
                if not GEMINI_API_KEY:
                    st.error("GEMINI_API_KEY олдсонгүй.")
                else:
                    genai.configure(api_key=GEMINI_API_KEY)
                    model = genai.GenerativeModel("gemini-2.5-flash")
                    gemini_prompt = f"""
Сурагчийн Bloom анализ дээр үндэслэн
Монгол хэлээр дараах бүтэцтэй зөвлөмж өг.

1. Давуу тал
2. Сул тал
3. Сайжруулах арга
4. Багшид өгөх зөвлөгөө

Bloom анализ:

{bloom_summary}
"""
                    response = model.generate_content(gemini_prompt)
                    gemini_text = response.text
                    gemini_html = html_lib.escape(gemini_text).replace("\n", "<br>")

                    st.session_state.gemini_result = gemini_text

                    if not st.session_state.final_ai_recommendation:
                        st.session_state.final_ai_recommendation = gemini_text

                    st.markdown(f"<div class='ai-card'>{gemini_html}</div>", unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Gemini error: {e}")

        with col_chatgpt:
            st.markdown("# ChatGPT AI")
            try:
                if not OPENAI_API_KEY:
                    st.error("OPENAI_API_KEY олдсонгүй.")
                else:
                    client = OpenAI(api_key=OPENAI_API_KEY)
                    response = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
                            {"role": "system", "content": "Монгол хэлээр боловсролын AI зөвлөмж гарга."},
                            {"role": "user", "content": prompt},
                        ],
                    )
                    text = response.choices[0].message.content
                    text_html = html_lib.escape(text).replace("\n", "<br>")

                    st.session_state.chatgpt_result = text
                    st.session_state.final_ai_recommendation = text

                    st.markdown(f"<div class='ai-card'>{text_html}</div>", unsafe_allow_html=True)
            except Exception as e:
                st.error(f"OpenAI error: {e}")

    st.subheader("Parent PDF-д оруулах AI зөвлөмж сонгох")

    ai_source_for_pdf = st.radio(
        "AI зөвлөмжийн эх сурвалж",
        [
            "ChatGPT AI",
            "Gemini AI",
            "Gemini + ChatGPT хамтад нь"
        ],
        horizontal=True,
        key="ai_source_for_pdf"
    )

    def get_selected_ai_text():
        if ai_source_for_pdf == "ChatGPT AI":
            return st.session_state.get("chatgpt_result", "")

        if ai_source_for_pdf == "Gemini AI":
            return st.session_state.get("gemini_result", "")

        gemini_part = st.session_state.get("gemini_result", "")
        chatgpt_part = st.session_state.get("chatgpt_result", "")

        return f"""
Gemini AI зөвлөмж:

{gemini_part if gemini_part else "Gemini AI зөвлөмж үүсээгүй байна."}

----------------------------------------

ChatGPT AI зөвлөмж:

{chatgpt_part if chatgpt_part else "ChatGPT AI зөвлөмж үүсээгүй байна."}
"""

    selected_ai_text = get_selected_ai_text()

    default_parent_text = make_parent_report_text(
        student_code=student_code,
        student_name=student_name,
        score=total_score,
        total=len(results),
        percent=percent,
        bloom_stats=bloom_stats,
        ai_text=selected_ai_text,
    )

    if st.button("🔄 Сонгосон AI зөвлөмжийг Parent PDF-д оруулах"):
        st.session_state.parent_report_text = default_parent_text
        st.rerun()

    if st.session_state.parent_report_text == "":
        st.session_state.parent_report_text = default_parent_text

    parent_review_text = st.text_area(
        "Parent Report Text",
        value=st.session_state.parent_report_text,
        height=650,
        label_visibility="collapsed",
    )

    st.session_state.parent_report_text = parent_review_text
    parent_pdf = generate_parent_report_pdf(parent_review_text)

    st.download_button(
        label="📥 Зассан Parent Report PDF татах",
        data=parent_pdf,
        file_name=f"{student_code}_{student_name}_Parent_Report.pdf",
        mime="application/pdf",
    )

    st.subheader("8. Add to Batch")

    if st.button("ADD TO BATCH"):
        exists = any(x["code"] == student_code for x in st.session_state.batch_results)
        if exists:
            st.warning("Энэ сурагч batch list-д байна.")
        else:
            st.session_state.batch_results.append({"code": student_code, "name": student_name, "score": percent})
            st.success("Batch list-д нэмэгдлээ.")


    st.subheader("9. Batch List / LXP Fill")

    # ============================================
    # LXP EXCEL MAPPING
    # ============================================

    with st.expander("📁 LXP Batch Excel Mapping", expanded=False):

        lxp_excel = st.file_uploader(
            "LXP batch excel upload",
            type=["xlsx"],
            key="lxp_batch_excel"
        )

        if lxp_excel is not None:

            lxp_df_original = pd.read_excel(lxp_excel)

            st.success("Excel амжилттай уншигдлаа")

            st.dataframe(
                lxp_df_original.head(),
                use_container_width=True
            )

            cols = list(lxp_df_original.columns)

            col_map1, col_map2, col_map3 = st.columns(3)

            with col_map1:
                code_col = st.selectbox(
                    "Student code column",
                    cols,
                    key="lxp_code_col"
                )

            with col_map2:
                name_col = st.selectbox(
                    "Student name column",
                    cols,
                    key="lxp_name_col"
                )

            with col_map3:
                score_col = st.selectbox(
                    "Score column",
                    cols,
                    key="lxp_score_col"
                )

            lxp_ready_df = pd.DataFrame({
                "StudentCode": lxp_df_original[code_col],
                "StudentName": lxp_df_original[name_col],
                "Score": lxp_df_original[score_col]
            })

            st.markdown("### ✅ LXP Ready Preview")

            st.dataframe(
                lxp_ready_df,
                use_container_width=True
            )

            if st.button("➕ Excel-с Batch List рүү нэмэх", key="add_excel_to_batch"):
                added_count = 0
                skipped_count = 0

                for _, row in lxp_ready_df.iterrows():
                    code = str(row["StudentCode"]).strip()
                    name = str(row["StudentName"]).strip()

                    try:
                        score = float(row["Score"])
                    except Exception:
                        score = 0

                    exists = any(
                        str(x.get("code", "")).strip() == code
                        for x in st.session_state.batch_results
                    )

                    if exists:
                        skipped_count += 1
                    else:
                        st.session_state.batch_results.append({
                            "code": code,
                            "name": name,
                            "score": score
                        })
                        added_count += 1

                st.success(
                    f"{added_count} сурагч Batch List-д нэмэгдлээ. "
                    f"{skipped_count} давхардсан сурагч алгасагдлаа."
                )
                st.rerun()


            output = BytesIO()

            with pd.ExcelWriter(output, engine="openpyxl") as writer:
                lxp_ready_df.to_excel(
                    writer,
                    index=False
                )

            st.download_button(
                "⬇️ Download LXP Ready Excel",
                data=output.getvalue(),
                file_name="LXP_READY.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

    # ============================================
    # CURRENT BATCH LIST
    # ============================================

    if len(st.session_state.batch_results) > 0:
        batch_df = pd.DataFrame(st.session_state.batch_results)
        st.dataframe(batch_df, use_container_width=True, hide_index=True)

        payload_json = json.dumps(st.session_state.batch_results, ensure_ascii=False)
        components.html(
            f"""
<script>
const LXP_PAYLOAD = {payload_json};

function postToExtension(action) {{
    const msg = {{
        source: "TEACHER_AI",
        action: action,
        payload: LXP_PAYLOAD
    }};

    try {{ window.postMessage(msg, "*"); }} catch(e) {{}}
    try {{ window.parent.postMessage(msg, "*"); }} catch(e) {{}}
    try {{ window.top.postMessage(msg, "*"); }} catch(e) {{}}
}}

postToExtension("SAVE_LXP_PAYLOAD");
setTimeout(() => postToExtension("SAVE_LXP_PAYLOAD"), 500);
setTimeout(() => postToExtension("SAVE_LXP_PAYLOAD"), 1500);
setTimeout(() => postToExtension("SAVE_LXP_PAYLOAD"), 3000);
</script>
<button onclick="sendAllToLXP()"
    style="
        background:#16a085;
        color:white;
        border:none;
        padding:14px 22px;
        border-radius:12px;
        font-weight:bold;
        cursor:pointer;
        font-size:16px;
        width:100%;
    ">
    SEND ALL TO LXP
</button>

<script>
async function sendAllToLXP(){{
    const textPayload = JSON.stringify(LXP_PAYLOAD);

    try {{
        await navigator.clipboard.writeText(textPayload);
    }} catch(e) {{
        console.log("Clipboard copy failed", e);
    }}

    postToExtension("SEND_TO_LXP");
    alert("Batch list clipboard-д хадгалагдлаа. Одоо LXP tab дээр extension → AUTO FILL LXP дарна.");
}}
</script>
""",
            height=80,
        )

        if st.button("CLEAR BATCH"):
            st.session_state.batch_results = []
            st.rerun()
    else:
        st.info("Batch list хоосон байна.")
elif app_mode == "📷 OMR шалгалт засах → LXP":
    st.info("Excel болон сурагчийн зураг оруулна уу.")
