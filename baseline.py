import streamlit as st
from pathlib import Path
import PyPDF2
from transformers import BigBirdPegasusForConditionalGeneration, AutoTokenizer

# # Initialize tokenizer and model
st.cache_data()
def get_models():
    tokenizer = AutoTokenizer.from_pretrained("bigbird-pegasus-large-pubmed-tokenizer")
    model = BigBirdPegasusForConditionalGeneration.from_pretrained("bigbird-pegasus-large-pubmed-model")
    return tokenizer, model

# Your existing functions here (extract_results_section, generate_summary, write_summaries_to_file)

tokenizer, model = get_models()

def main():
    # Streamlit app
    st.title("PDF Results Section Extractor and Summarizer")

    # File uploader allows user to add files
    uploaded_file = st.file_uploader("Choose a PDF file")

    if uploaded_file is not None:
        st.write("File uploaded successfully.")
            # To read file as bytes:
        bytes_data = uploaded_file.getvalue()
        process_uploaded_file(bytes_data)

def process_uploaded_file(bytes_data):
    st.write("process_uploaded_file")
    with open("temp_file.pdf", "wb") as f:
        f.write(bytes_data)
    file_path = Path("temp_file.pdf")
    result = extract_results_section(file_path)
    if result:
        summary = generate_summary(result)
        st.write("Summary:")
        st.write(summary)
    else:
        st.write("Unable to extract or summarize the results section.")

def extract_results_section(file_path):
    print("extract_results_section")
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        number_pages = len(reader.pages)
        extracting = False
        results_text = ""

        for page_number in range(1, number_pages):
            page = reader.pages[page_number]
            text = page.extract_text()
            if text:
                if not extracting:
                    if "Results" in text or "RESULTS" in text:
                        extracting = True
                        results_text += text
                else:
                    if "Discussion" in text or "DISCUSSION" in text or "Conclusion" in text or "CONCLUSION" in text:
                        break
                    results_text += text

        if not results_text:
            return None
    return results_text

def generate_summary(results_text):
    print("generate_summary")
    # Pass the inputs to the BigBirdPegasus model and tokenizer
    inputs = tokenizer(results_text, return_tensors='pt')
    prediction = model.generate(**inputs, max_length=550, length_penalty=0.9, early_stopping=True)
    summary = tokenizer.batch_decode(prediction, skip_special_tokens=True)
    return " ".join(summary)

main()
