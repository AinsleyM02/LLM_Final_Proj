from bert_score import score
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Define the ground truth based on trusted sources (e.g., NIH, medical textbooks)


ground_truth = [
    """
- **Overview**:  
Atrial fibrillation (AF) is the most common type of cardiac arrhythmia, characterized by abnormal electrical activity in the atria of the heart, causing them to fibrillate. It is a tachyarrhythmia, typically resulting in a fast and irregular heart rate. AF is a leading cause of stroke and is often associated with various cardiovascular conditions, advancing age, and other health issues like hypertension and alcohol consumption.

- **Presentation and Symptoms**:  
Symptoms of atrial fibrillation can vary from asymptomatic to more severe manifestations, including palpitations, chest pain, shortness of breath, dizziness, fatigue, nausea, and diaphoresis (sweating). Some patients may experience a rapid heart rate, while others may not notice any symptoms. It is also possible for AF to cause heart failure or even stroke due to its turbulent blood flow and clot formation in the heart.

- **Pathophysiology**:  
Atrial fibrillation primarily results from structural and electrical changes in the atria, commonly caused by cardiac remodeling. These changes may include fibrosis, altered myocyte function, and irregular electrical firing from ectopic foci, often near the pulmonary veins. The rapid and irregular electrical impulses disrupt normal heart rhythm, impairing blood flow, and increasing the likelihood of thrombus formation, which can lead to stroke if dislodged.

- **Diagnosis**:  
Diagnosis of atrial fibrillation is typically made through an electrocardiogram (ECG), which reveals the characteristic irregularly irregular rhythm and absent P-waves. Additional diagnostic tests may include blood work (e.g., thyroid function, CBC), imaging (e.g., chest X-ray), and transesophageal echocardiography (TEE) to assess for blood clots and structural heart issues.

- **Treatment**:  
Treatment aims to control symptoms, reduce the risk of stroke, and manage underlying causes. Options include:
  - **Rate control** (e.g., beta-blockers, calcium channel blockers, digoxin)
  - **Rhythm control** (e.g., amiodarone)
  - **Anticoagulation** to reduce stroke risk (e.g., non-vitamin K oral anticoagulants like apixaban)
  - **Cardioversion** (electrical or pharmacologic)
  - **Ablation therapy** and **pacemaker implantation** in severe cases

- **Complications**:  
The most significant complication of atrial fibrillation is the increased risk of stroke due to clot formation in the atria, which can embolize to the brain. Other potential complications include heart failure, cardiomyopathy, and long-term anticoagulation-related bleeding issues. Management aims to reduce these risks through appropriate therapy and monitoring.
    """

]

# Responses from the base model (provided earlier)
base_model_output = [
    """
Atrial Fibrillation (AFib) is an irregular and often rapid heart rhythm that originates in the atria (the upper chambers of the heart). Here's  
what you need to know about this common heart condition:

1. Causes: The exact cause of AFib isn't always known, but age, high blood pressure, coronary artery disease, obesity, and other structural heart

problems can increase the risk.

2. Symptoms: Common symptoms include a rapid, irregular pulse; palpitations (sensations of a racing, fluttering, or pounding heart); fatigue;    
shortness of breath; and dizziness or lightheadedness. Some people may not experience any symptoms at all, especially if the condition is mild.

1. Diagnosis: AFib can be diagnosed through an electrocardiogram (ECG), which records the electrical activity of the heart. Other tests,
such as echocardiography or holter monitoring, may also be used to confirm the diagnosis and assess the condition's severity.

2. Complications: AFib can increase the risk of stroke, heart failure, and other complications like congestive heart failure, blood clots,       
and decreased cardiac output.

3. Treatment: Treatment options for AFib include medications such as blood thinners to reduce the risk of stroke, rate control drugs to 
slow the heart rate, rhythm control drugs to restore a normal heart rhythm, and catheter ablation procedures to create scar tissue in the        
atria to block the abnormal electrical signals. Lifestyle changes like maintaining a healthy weight, regular exercise, quitting smoking,         
and managing stress can also help manage AFib.

4. Prevention: Maintaining a healthy lifestyle, managing high blood pressure and other risk factors, and undergoing appropriate medical
treatment can help prevent or reduce the risk of developing AFib. Regular check-ups with a healthcare provider are essential for those at        
higher risk.
Atrial Fibrillation (AFib) is an irregular and often rapid heart rhythm that originates in the atria (the upper chambers of the heart). Here's  
what you need to know about this common heart condition:

1. Causes: The exact cause of AFib isn't always known, but age, high blood pressure, coronary artery disease, obesity, and other structural heart

problems can increase the risk.

2. Symptoms: Common symptoms include a rapid, irregular pulse; palpitations (sensations of a racing, fluttering, or pounding heart); fatigue;    
shortness of breath; and dizziness or lightheadedness. Some people may not experience any symptoms at all, especially if the condition is mild.
    """
]

# Responses from the RAG-based model (custom model with the database)
rag_model_output = [
    """
*Overview**: Atrial Fibrillation (AF) is an irregular and often rapid heart rhythm that originates from the atria, the upper chambers of the heart. It is a common heart condition, particularly among older adults, and can lead to complications such as stroke if left untreated.

**Presentation and Symptoms**: AF may present with symptoms like palpitations, shortness of breath, dizziness, weakness, and fatigue. Some individuals may not experience any noticeable symptoms at all.

**Pathophysiology**: The primary causes of AF are hypertension and atherosclerotic heart disease, with other potential causes including various factors such as age, obesity, alcohol consumption, and genetics. The condition can be remembered using the mnemonic "I SMART CHAP".

**Diagnosis**: AF is typically diagnosed through an electrocardiogram (ECG) or a 24-hour ambulatory ECG monitoring. Further tests may include echocardiography, chest X-ray, or blood tests to rule out other conditions.

**Treatment**: Acute AF can be treated with direct current cardioversion if the patient is unstable. If stable, initial management involves ventricular rate control using atrioventricular nodal-blocking agents such as digoxin, beta-blockers, diltiazem, or verapamil. Chronic AF requires long-term anticoagulation to prevent embolic strokes; however, "lone atrial fibrillation," which is associated with a lower risk of stroke, may not require anticoagulation.

**Complications**: Untreated or inadequately managed AF can lead to complications like heart failure, stroke, and death. Regular monitoring and long-term management are crucial for preventing these complications.
    """
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
