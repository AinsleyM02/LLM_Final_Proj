from bert_score import score
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Define the ground truth based on trusted sources (e.g., NIH, medical textbooks)
#ground_truth = [
#    "Diabetes occurs when your blood glucose, also called blood sugar, is too high. High blood glucose can cause health problems over time. The main types of diabetes are type 1, type 2, and gestational. Symptoms and causes: Increased thirst and urination, feeling tired, unexplained weight loss, and blurred vision are symptoms of diabetes. Many people have no symptoms and don’t know they have diabetes. Each type of diabetes has different causes. The main signs of diabetes mellitus include increased thirst, frequent urination, increased hunger, weight loss despite eating more, fatigue, blurred vision, slow-healing sores, and frequent infections. Over time, high blood sugar levels can lead to complications such as nerve damage, kidney damage, heart disease, and vision loss. Diabetes Mellitus occurs when the body either doesn't produce enough insulin or becomes resistant to its effects. Insulin is a hormone produced by the pancreas that allows cells in the body to absorb glucose (sugar) from the blood for energy. In diabetes, this process is impaired, leading to high blood sugar levels. A diagnosis of diabetes mellitus can be made through a combination of tests, such as fasting blood glucose tests, oral glucose tolerance tests, and hemoglobin A1C tests. These tests measure the amount of glucose in the blood and provide an indication of average blood sugar levels over time. Treatment for diabetes mellitus involves managing blood sugar levels through a combination of medications, diet, exercise, and lifestyle changes. Medications used to control blood sugar levels include insulin, oral antidiabetic drugs, and other agents that help the body use insulin more effectively. If left untreated or poorly managed, diabetes mellitus can lead to a range of complications, such as nerve damage (neuropathy), kidney damage (nephropathy), vision loss (retinopathy), heart disease, and foot problems that may result in amputation."
#]

ground_truth = [
    """
- **Overview**: Type 2 Diabetes Mellitus (T2DM) is a chronic metabolic disease characterized by elevated blood glucose levels resulting from either defective insulin secretion or impaired insulin action (insulin resistance). It is primarily caused by poor lifestyle choices, including diet and physical inactivity, and typically affects middle-aged or older adults. T2DM is one of the leading causes of death globally.

- **Presentation and Symptoms**: Common symptoms of T2DM include polyuria (frequent urination), polydipsia (increased thirst), fatigue, weight loss, and blurred vision. Patients may also experience acanthosis nigricans (dark, velvety patches of skin), particularly in areas like the neck or armpits. In advanced cases, neuropathic pain, frequent infections, and blurry vision may develop.

- **Pathophysiology**: T2DM involves insulin resistance, where the body's cells fail to respond adequately to insulin, and later a reduced ability of the pancreas to secrete insulin. This results in elevated blood glucose levels. The disease is influenced by both genetic factors (with a higher risk among those with a family history) and environmental factors like obesity and physical inactivity. Chronic hyperglycemia can damage small blood vessels in the eyes, kidneys, and nerves, leading to diabetic retinopathy, nephropathy, and neuropathy.

- **Diagnosis**: T2DM is diagnosed based on criteria such as a fasting plasma glucose level ≥126 mg/dL, an HbA1c ≥6.5%, or a 2-hour plasma glucose level ≥200 mg/dL during an oral glucose tolerance test (OGTT). The condition is often diagnosed through routine screening, especially for individuals aged 45 and older or those who are overweight.

- **Treatment**: Initial treatment for T2DM involves lifestyle modifications, such as dietary changes (low-carb, calorie restriction) and increased physical activity. Medications like metformin, which improve insulin sensitivity, are commonly prescribed. Other treatments include sulfonylureas, DPP-4 inhibitors, GLP-1 receptor agonists, and SGLT-2 inhibitors. In advanced cases, insulin therapy may be necessary. Bariatric surgery may be an option for morbidly obese individuals.

- **Complications**: Long-term complications of poorly controlled T2DM include cardiovascular disease (ASCVD), diabetic retinopathy (leading to blindness), diabetic nephropathy (leading to end-stage renal disease), and diabetic neuropathy (leading to foot ulcers and amputation). Diabetic patients are also at higher risk for infections, particularly in the urinary tract and skin. The risk of these complications increases with the duration of the disease and poor glucose control. Additionally, individuals with T2DM are at higher risk for certain cancers, including bladder cancer.
    """
]
# Responses from the base model (provided earlier)
base_model_output = [
"Diabetes Mellitus Type 2, often referred to as Type 2 Diabetes, is a long-term metabolic disorder characterized by high blood sugar levels due to the body's ineffective use of insulin or inadequate insulin production. Here's an overview of what you need to know: 1. Causes: The exact cause of Type 2 Diabetes isn't known, but it's associated with genetic, environmental, and lifestyle factors. Obesity, physical inactivity, and a family history of diabetes are significant risk factors. As the body becomes resistant to insulin or can't produce enough insulin, glucose levels in the blood increase. 2. Symptoms: Common symptoms include frequent urination, increased thirst, increased hunger, fatigue, blurred vision, slow-healing sores, and frequent infections. Some individuals may not exhibit symptoms until complications arise. 3. Diagnosis: A doctor can diagnose Type 2 Diabetes through a series of tests, including fasting plasma glucose test, oral glucose tolerance test, or HbA1c (a blood test that measures your average blood sugar levels over the past 3 months). 4. Complications: Untreated or poorly managed diabetes can lead to complications like heart disease, stroke, kidney damage, nerve damage, and vision loss. 5. Treatment: Lifestyle changes such as regular exercise, healthy eating, and maintaining a healthy weight are essential for managing Type 2 Diabetes. Medications like metformin or insulin may also be prescribed to help manage blood sugar levels. In some cases, surgery (such as bariatric surgery) might be an option. 6. Prevention: Maintaining a healthy lifestyle can help prevent the onset of Type 2 Diabetes. This includes regular exercise, a balanced diet, and maintaining a healthy weight. If you have a family history of diabetes, regular screenings are recommended."
]

# Responses from the RAG-based model (custom model with the database)
rag_model_output = [
"Overview: Diabetes Mellitus is a chronic metabolic disorder characterized by high levels of glucose (sugar) in the blood due to issues with insulin production or function. It affects various body parts, leading to symptoms such as increased thirst, frequent urination, fatigue, and blurred vision. Presentation and Symptoms: The main signs of diabetes mellitus include increased thirst, frequent urination, increased hunger, weight loss despite eating more, fatigue, blurred vision, slow-healing sores, and frequent infections. Over time, high blood sugar levels can lead to complications such as nerve damage, kidney damage, heart disease, and vision loss. Pathophysiology: Diabetes Mellitus occurs when the body either doesn't produce enough insulin or becomes resistant to its effects. Insulin is a hormone produced by the pancreas that allows cells in the body to absorb glucose (sugar) from the blood for energy. In diabetes, this process is impaired, leading to high blood sugar levels. Diagnosis: A diagnosis of diabetes mellitus can be made through a combination of tests, such as fasting blood glucose tests, oral glucose tolerance tests, and hemoglobin A1C tests. These tests measure the amount of glucose in the blood and provide an indication of average blood sugar levels over time. Treatment: Treatment for diabetes mellitus involves managing blood sugar levels through a combination of medications, diet, exercise, and lifestyle changes. Medications used to control blood sugar levels include insulin, oral antidiabetic drugs, and other agents that help the body use insulin more effectively. Complications: If left untreated or poorly managed, diabetes mellitus can lead to a range of complications, such as nerve damage (neuropathy), kidney damage (nephropathy), vision loss (retinopathy), heart disease, and foot problems that may result in amputation."
    ]

# Compute BERTScore
def compute_bertscore(candidate_texts, reference_texts):
    """
    Compute BERT score for a candidate sentence against a reference sentence.
    """
    P, R, F1 = score(candidate_texts, reference_texts, lang="en")
    return F1.mean().item(), P.mean().item(), R.mean().item()

# Evaluate BERTScore for the Base Model
f1_base, p_base, r_base = compute_bertscore(base_model_output, ground_truth)
print(f"Base Model BERTScore F1: {f1_base}, Precision: {p_base}, Recall: {r_base}")

# Evaluate BERTScore for the RAG-based Model
f1_rag, p_rag, r_rag = compute_bertscore(rag_model_output, ground_truth)
print(f"RAG Model BERTScore F1: {f1_rag}, Precision: {p_rag}, Recall: {r_rag}")

# Initialize GPT-2 for Perplexity calculation
gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2")
gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
gpt2_model.eval()

def compute_perplexity(text, model, tokenizer):
    """
    Compute the perplexity of a given text using GPT-2.
    """
    inputs = tokenizer.encode(text, return_tensors="pt")
    input_ids = inputs

    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        log_likelihood = outputs.loss

    perplexity = torch.exp(log_likelihood)
    return perplexity.item()

# Example for Perplexity evaluation on the Base Model output
perplexity_base = compute_perplexity(base_model_output[0], gpt2_model, gpt2_tokenizer)
print(f"Perplexity of the Base Model output: {perplexity_base}")

# Example for Perplexity evaluation on the RAG Model output
perplexity_rag = compute_perplexity(rag_model_output[0], gpt2_model, gpt2_tokenizer)
print(f"Perplexity of the RAG Model output: {perplexity_rag}")
