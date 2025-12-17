import gradio as gr
import base64

from main import (
    load_docs,
    split_docs,
    vector_store,
    build_QA_pipeline,
    list_sensor_files,
    load_sensor_file,
    interpret_sensor_data,
    build_query_with_data
)

# ============================================================
#            BUILD RAG PIPELINE ONCE (FAST STARTUP)
# ============================================================

documents = load_docs("test_docs")
chunks = split_docs(documents)
print("Number of chunks:", len(chunks))
vs = vector_store(chunks)
qa_pipeline = build_QA_pipeline(vs)


def rag_answer(user_question, selected_sensor, use_sensors):
    if use_sensors:
        sensor_data = load_sensor_file(selected_sensor)
        sensor_summary = interpret_sensor_data(sensor_data)
    else:
        sensor_summary = None

    query = build_query_with_data(sensor_summary, user_question)

    result = qa_pipeline.invoke({"input": query})
    answer = result["answer"]

    sources_text = ""
    for i, doc in enumerate(result["context"], start=1):
        excerpt = doc.page_content[:150].replace("\n", " ")
        sources_text += f"Source {i}: {doc.metadata.get('source')} — \"{excerpt}...\"\n\n"

    return sensor_summary, answer, sources_text



# ============================================================
#                CHICKEN COOP COMFORT UI LOGIC
# ============================================================

chat_history = []  # Store (user_msg, bot_msg)


def format_sensor_cards(sensor_summary: str) -> str:
    """Convert the sensor summary into pretty color-coded HTML cards."""
    lines = sensor_summary.split("\n")
    html = "<div style='display:flex; flex-direction:column; gap:8px;'>"

    for line in lines:
        color = "#EEE"
        if line.startswith("✔️"):
            color = "#D9F7BE"
        elif line.startswith("⚠️"):
            color = "#FFF1B8"
        elif line.startswith("❗"):
            color = "#FFA39E"

        html += f"""
        <div style='background:{color}; padding:10px; border-radius:8px; font-size:15px;'>
            {line}
        </div>
        """

    html += "</div>"
    return html


def rag_answer_for_ui(user_question, selected_sensor, use_sensors):

    if use_sensors:
        sensor_data = load_sensor_file(selected_sensor)
        sensor_summary = interpret_sensor_data(sensor_data)
        sensor_cards = format_sensor_cards(sensor_summary)
    else:
        sensor_summary = None
        sensor_cards = "<div style='color:#888;'>Sensor data is disabled.</div>"

    _, answer, sources = rag_answer(user_question, selected_sensor, use_sensors)

    chat_history.append((user_question, answer))
    chat_md = ""
    chat_md = ""
    for user, bot in chat_history:
        chat_md += (
        "### 👤 You\n"
        f"{user.strip()}\n\n"
        "### 🤖 ChatKippieT\n"
        f"{bot.strip()}\n\n"
        "---\n\n"
    )
    return sensor_cards, chat_md, sources


# ============================================================
#                      IMAGE HELPERS
# ============================================================

def encode_image(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


logo_b64 = encode_image("assets/chicken_logo.png")
bert_b64 = encode_image("assets/bert.png")
bella_b64 = encode_image("assets/bella.png")
nugget_b64 = encode_image("assets/nugget.png")


# ============================================================
#                      GRADIO APP LAYOUT
# ============================================================

with gr.Blocks(title="🐔 Chicken Coop Comfort") as demo:

    # ---------- GLOBAL STYLING ----------
    gr.HTML("""
    <style>
        body {
            background-color: #f5f1e8 !important;
        }

        /* Ask button styling */
        #ask_btn {
            background-color: #da8242 !important;
            color: white !important;
            font-size: 18px !important;
            border-radius: 10px !important;
            padding: 12px 25px !important;
        }

        /* FIXED LOGO */
        #logo_container {
            position: fixed !important;
            top: 15px;
            right: 20px;
            z-index: 99999;
            pointer-events: none;
        }

        #logo_container img {
            height: 150px;
        }

        /* Force correct fixed behavior */
        body > #logo_container {
            position: fixed !important;
        }

        /* Header styling */
        .ccc_header {
            text-align: center;
            padding: 15px;
            background-color: #f5f1e8;
            border-radius: 12px;
            border: 2px solid #da8242;
            margin-bottom: 20px;
            max-width: 90%;
            margin-left: auto;
            margin-right: auto;
        }

        .ccc_title {
            color: #da8242;
            margin-bottom: 5px;
            font-size: 38px;
            font-weight: bold;
        }

        .ccc_slogan {
            color: #6A4E23;
            margin-top: 0;
            font-size: 20px;
        }

        /* Scrollable chat area */
        #chat_display {
            overflow-y: visible;
            padding-right: 8px;
        }

        /* Scrollable sources */
        #sources_box {
            max-height: 240px;
            overflow-y: auto;
        }

        /* Chicken friend cards */
        .chicken-card-row {
            display: flex;
            flex-wrap: wrap;
            gap: 20px;
            justify-content: center;
            margin-top: 30px;
            margin-bottom: 30px;
        }

        .chicken-card {
            width: 200px;
            height: 200px;
            perspective: 1000px;
        }

        .card-inner {
            position: relative;
            width: 100%;
            height: 100%;
            transition: transform 0.6s;
            transform-style: preserve-3d;
            cursor: pointer;
        }

        .chicken-card:hover .card-inner {
            transform: rotateY(180deg);
        }

        .card-front, .card-back {
            position: absolute;
            width: 100%;
            height: 100%;
            backface-visibility: hidden;
            border-radius: 20px;
            border: 2px solid #da8242;
            background: #fffdf7;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            padding: 10px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.05);
        }

        .card-front img {
            max-width: 80%;
            max-height: 70%;
            margin-bottom: 8px;
        }

        .card-front h4, .card-back p {
            margin: 0;
            text-align: center;
            font-size: 14px;
            color: #6A4E23;
        }

        .card-back {
            transform: rotateY(180deg);
        }
    </style>
    """)

    # ---------- FIXED LOGO ----------
    gr.HTML(f"""
        <div id="logo_container">
            <img src="data:image/png;base64,{logo_b64}" alt="Chicken Coop Comfort Logo">
        </div>
    """)

    # ---------- HEADER ----------
    gr.HTML("""
        <div class="ccc_header">
            <h1 class="ccc_title">🐔 Chicken Coop Comfort</h1>
            <h3 class="ccc_slogan">Worry-Free Chicken Keeping</h3>
        </div>
    """)

    # ---------- MAIN LAYOUT ----------
    with gr.Row():

        # LEFT controls
        with gr.Column(scale=1):

            use_sensors = gr.Checkbox(
                label="Use Real-Time Sensor Data",
                value=True
            )

            sensor_explanation = gr.Markdown(
                " **Sensor Mode**: When enabled, The system uses real sensor readings in your answers. "
                "Turn off to ask general chicken-keeping questions without live data."
            )

            sensor_choice = gr.Dropdown(
            choices=list_sensor_files("Data/sensors"),
            label="Select Sensor Dataset",
            value="sensor_data_1.json"
            )

            question = gr.Textbox(
            label="Ask a question about your chickens",
            lines=2,
            placeholder="Example: Why is my chicken panting?"
            )

            ask_btn = gr.Button("Ask", elem_id="ask_btn")
            sensor_cards_display = gr.HTML(label="📡 Sensor Status")



        # RIGHT chat + sources
        with gr.Column(scale=2):
            chat_display = gr.Markdown(label="💬 Chat History", elem_id="chat_display")

            with gr.Accordion("📚 Sources Used", open=False):
                sources_box = gr.Textbox(
                    label="",
                    lines=6,
                    elem_id="sources_box"
                )

    # ---------- CHICKEN CHARACTER CARDS ----------
    gr.HTML(f"""
    <div class="chicken-card-row">

      <div class="chicken-card">
        <div class="card-inner">
          <div class="card-front">
            <img src="data:image/png;base64,{bert_b64}" alt="Bert" />
            <h4>Bert</h4>
          </div>
          <div class="card-back">
            <p><b>Bert</b> loves eating. In the mornings he's shy, but once the sun is out he really grabs the day!</p>
          </div>
        </div>
      </div>

      <div class="chicken-card">
        <div class="card-inner">
          <div class="card-front">
            <img src="data:image/png;base64,{bella_b64}" alt="Bella" />
            <h4>Bella</h4>
          </div>
          <div class="card-back">
            <p><b>Bella</b> is sweet and curious. She explores every corner of the coop.</p>
          </div>
        </div>
      </div>

      <div class="chicken-card">
        <div class="card-inner">
          <div class="card-front">
            <img src="data:image/png;base64,{nugget_b64}" alt="Nugget" />
            <h4>Nugget</h4>
          </div>
          <div class="card-back">
            <p><b>Nugget</b> is adventurous (please close the gate!) but always happy to see you.</p>
          </div>
        </div>
      </div>

    </div>
    """)

    # ---------- BUTTON ACTION ----------
    ask_btn.click(
    rag_answer_for_ui,
    inputs=[question, sensor_choice, use_sensors],
    outputs=[sensor_cards_display, chat_display, sources_box]
)

demo.launch()