#Starry Night Single File
#pip install PyPDF2
#pip install transformers    
#pip install torch
#pip install pathlib

from pathlib import Path
import pyPDF2
from transformers import BigBirdPegasusForConditionalGeneration, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("google/bigbird-pegasus-large-pubmed")
model = BigBirdPegasusForConditionalGeneration.from_pretrained("google/bigbird-pegasus-large-pubmed")

'''HERE IS THE FUNCTIONS OF THE PATH'''

def process_path(file_path, summary_file_path='all_summaries.txt'):
    path = Path(file_path)
    all_summaries = []
    
    if path.is_file() and path.suffix.lower() == '.pdf':
        # Process a single PDF file
        result = extract_results_section(path)
        if result:
            summary = generate_summary(result)
            all_summaries.append(summary)
            write_summaries_to_file(all_summaries, summary_file_path) 
        else:
            return "Unable to extract or summarize the results section."
            
    elif path.is_dir():
        # Process each PDF file in the directory
        for file_path in path.iterdir():
            if file_path.is_file() and file_path.suffix.lower() == '.pdf':
                print("Processing PDF file:", file_path.name)
                result = extract_results_section(file_path)
                if result:
                    summary = generate_summary(result)
                    all_summaries.append(summary)  
                else:
                    print("Unable to extract or summarize the results section for:", file_path.name)
            else:
                print("Skipping non-PDF file:", file_path.name)
            write_summaries_to_file(all_summaries, summary_file_path)
        return "Finished processing PDF files in the folder."
    
    else:
        return "The input path is neither a single PDF file nor a folder of PDF files. Please check and try again."
    


'''HERE ARE THE FUNCTIONS THAT WILL BE CALLED IN THE process_path FUNCTION'''

# Extract the Results section from a PDF file
def extract_results_section(file_path):
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

# Generate a summary of the Results section

def generate_summary(results_text):
    # Pass the inputs to the BigBirdPegasus model and tokenizer
    inputs = tokenizer(results_text, return_tensors='pt')
    prediction = model.generate(**inputs)
    summary = tokenizer.batch_decode(prediction, skip_special_tokens=True)
    return " ".join(summary)    
    

# Write the summary to a file

def write_summaries_to_file(summaries, file_path):
    with open(file_path, 'w') as file:
        for summary in summaries:
            file.write(summary + "\n\n") 









