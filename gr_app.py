import gradio as gr
from langsmith import traceable
from main import run_ocr

def process_file(file):
    """
    Process uploaded file through OCR system
    Args:
        file: Path to the uploaded file
    Returns:
        JSON formatted OCR results
    """
    return run_ocr(file_url_or_path=file, port=3203)

# Define example POD files for quick testing
EXAMPLE_FILES = [
    "./data/POD_17572186_JPLQ_FT.pdf",
    "./data/POD_17536410_FLOK_FP.pdf"
]

# Create Gradio interface with defined configuration
iface = gr.Interface(
    fn=process_file,
    inputs=[
        gr.File(
            type="filepath",
            label="Upload PDF"
        )
    ],
    outputs=gr.JSON(label="OCR Results"),
    title="Image Analysis with InternVL2",
    description="Upload an image and enter a prompt to get AI analysis",
    examples=EXAMPLE_FILES
)

if __name__ == "__main__":
    iface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True
    )