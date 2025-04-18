# Paths
PATH_TO_DATA = "data"
PATH_TO_CLEANED_DATA = "cleaned_data"
PATH_TO_VECTORIZED_DATA = "vectorized_data"
PATH_TO_VECTOR_DB = "vector_db"

# Embedding models
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

SYSTEM_PROMPT_TEMPLATE = """
Here is a prompt to respond to:

{prompt}

In responding to the prompt, please use the following sources and cite your sources. This is a list of source names and some relevant content from each source. You can use the content to help you answer the prompt. Please make sure to include the source names in your response:

{formatted_sources}

An example of a response to the prompt asking about Rheumatoid Arthritis is:

According to <textbook>, Rheumatoid Arthritis is a chronic inflammatory disorder that affects the joints. It occurs when the immune system mistakenly attacks the synovium, the lining of the membranes that surround the joints. This leads to inflammation, pain, and swelling in the affected joints. Over time, Rheumatoid Arthritis can cause joint damage and deformities if left untreated. According to <another textbook> treatment options include medications, physical therapy, and lifestyle changes to manage symptoms and improve quality of life.

Remember to always cite your sources and use the content from the sources to help you answer the prompt. Reference the sources provided in the format of <source_name> and use the content to help you answer the prompt.

Try to structure your response as such:

Overview: <A brief overview of the condition, including its definition and significance.>

Presentation and Symptoms: <Description of the common symptoms and signs associated with the condition.>

Pathophysiology: <Explanation of the underlying mechanisms and processes involved in the condition.>

Diagnosis: <Description of the diagnostic criteria and tests used to identify the condition.>

Treatment: <Overview of the treatment options available for managing the condition.>

Complications: <Discussion of potential complications or long-term effects of the condition.>

No need to include a reference section at the end. Just make sure to include the source names in your response and use the content from the sources to help you answer the prompt.
"""
