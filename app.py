import gradio as gr
from main import answer_with_realtime_data, list_sensor_files, load_sensor_file, interpret_sensor_data

def rag_answer(user_question, selected_sensor):
    # Load the selected sensor dataset
    sensor_data = load_sensor_file(selected_sensor)
    sensor_summary = interpret_sensor_data(sensor_data)

    # Run your RAG pipeline
    result = answer_with_realtime_data(
        folder_path="test_docs",
        json_path=f"Data/sensors/{selected_sensor}",
        user_question=user_question
    )

    answer = result["answer"]

    # Build the sources output text
    sources_text = ""
    for i, doc in enumerate(result["context"], start=1):
        sources_text += f"Source {i}: {doc.metadata.get('source')}\n"
        sources_text += doc.page_content[:500] + "\n\n"

    return sensor_summary, answer, sources_text


# ----------------------------
#      GRADIO UI
# ----------------------------

with gr.Blocks(title="🐓 Chicken RAG Assistant") as demo:

    gr.Markdown("## 🐓 Chicken RAG Assistant")
    gr.Markdown("AI assistant with document-based reasoning and real-time sensor interpretation.")

    # Select sensor dataset
    sensor_choice = gr.Dropdown(
        choices=list_sensor_files("Data/sensors"),
        label="Select Sensor Dataset",
        value="normal.json"
    )

    # User question input
    question = gr.Textbox(label="Ask a question about the chickens")

    # Output panels
    sensor_panel = gr.Textbox(label="📡 Sensor Health Status", lines=8, interactive=False)
    answer_box = gr.Textbox(label="AI Answer", lines=6)
    sources_box = gr.Textbox(label="Sources Used", lines=10)

    ask_btn = gr.Button("Ask")

    ask_btn.click(
        rag_answer,
        inputs=[question, sensor_choice],
        outputs=[sensor_panel, answer_box, sources_box]
    )

demo.launch()
