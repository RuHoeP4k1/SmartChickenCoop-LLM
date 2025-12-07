import gradio as gr
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


def rag_answer(user_question, selected_sensor):
    """Core backend RAG function (no UI formatting)."""

    # 1. Load the selected sensor dataset
    sensor_data = load_sensor_file(selected_sensor)
    sensor_summary = interpret_sensor_data(sensor_data)

    # 2. Build the final enriched query
    query = build_query_with_data(sensor_summary, user_question)

    # 3. Run RAG pipeline
    result = qa_pipeline.invoke({"input": query})

    # 4. Extract answer
    answer = result["answer"]

    # 5. Extract sources
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


def rag_answer_for_ui(user_question, selected_sensor):
    """UI wrapper around RAG to handle formatting & chat history."""

    # Sensor interpretation
    sensor_data = load_sensor_file(selected_sensor)
    sensor_summary = interpret_sensor_data(sensor_data)
    sensor_cards = format_sensor_cards(sensor_summary)

    # RAG backend
    sensor_raw, answer, sources = rag_answer(user_question, selected_sensor)

    # Update chat history
    chat_history.append((user_question, answer))

    # Build chat HTML
    chat_html = "<div style='display:flex; flex-direction:column; gap:10px;'>"
    for user, bot in chat_history:
        chat_html += f"""
        <div style='background:#FFF8E7; padding:10px; border-radius:8px;'>
            <b>👤 You:</b> {user}
        </div>
        <div style='background:#E8F5E9; padding:10px; border-radius:8px;'>
            <b>🤖 CoopBot:</b> {bot}
        </div>
        """
    chat_html += "</div>"

    return sensor_cards, chat_html, sources


# ============================================================
#                      GRADIO APP LAYOUT
# ============================================================

with gr.Blocks(title="🐔 Chicken Coop Comfort") as demo:

    # Page styling
    gr.HTML("""
        <style>
        body {
            background-color: #FFFDF7 !important;
        }
        #ask_btn {
            background-color: #D17A22 !important;
            color: white !important;
            font-size: 18px !important;
            border-radius: 10px !important;
            padding: 10px 20px !important;
        }
        </style>
    """)

    # Header
    gr.HTML("""
        <div style="text-align: center; padding: 20px;
                    background-color: #FFF8E7;
                    border-radius: 12px;
                    border: 2px solid #F4D06F;
                    margin-bottom: 20px;">

            <h1 style="color: #D17A22; margin-bottom: 10px; font-size: 38px;">
                🐔 Chicken Coop Comfort
            </h1>

            <h3 style="color: #6A4E23; margin-top: 0;">
                Smart Monitoring • Real-Time Insights • Happier Chickens
            </h3>

        </div>
    """)

    # Two-column layout
    with gr.Row():

        # LEFT -------------------------------------------------
        with gr.Column(scale=1):

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

        # RIGHT ------------------------------------------------
        with gr.Column(scale=2):

            chat_display = gr.HTML(label="💬 Chat History")
            sources_box = gr.Textbox(label="📚 Sources Used", lines=6)

    # Connect button
    ask_btn.click(
        rag_answer_for_ui,
        inputs=[question, sensor_choice],
        outputs=[sensor_cards_display, chat_display, sources_box]
    )


demo.launch()
