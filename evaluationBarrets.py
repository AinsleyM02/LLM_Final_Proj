from bert_score import score
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Define the ground truth based on trusted sources (e.g., NIH, medical textbooks)


ground_truth = [
    """
**Overview**: Barrett esophagus is a premalignant condition where the normal esophageal squamous epithelium is replaced by columnar epithelium with goblet cells, indicative of intestinal metaplasia. It often develops as a complication of long-standing gastroesophageal reflux disease (GERD) and increases the risk of esophageal adenocarcinoma, a highly aggressive cancer. The condition is found in 5% to 12% of patients with chronic GERD and has a known association with certain genetic and lifestyle factors.

**Presentation and Symptoms**: Most patients with Barrett esophagus have symptoms of GERD, such as heartburn, acid regurgitation, and dysphagia. Less common symptoms include chest pain, sore throat, hoarseness, chronic cough, melena, or weight loss. Some individuals may be asymptomatic. The presence of symptoms often correlates with the severity of GERD.

**Pathophysiology**: Barrett esophagus is believed to result from chronic acid exposure due to GERD, leading to the replacement of squamous epithelium with columnar epithelium in the lower esophagus. This metaplastic change is a protective response to the acidity, but it increases the risk of further malignant transformation. Inflammatory cytokines and bile acids contribute to this process, and mutations in genes such as p16, CDX2, and TP53 are commonly found in the affected tissue.

**Diagnosis**: Diagnosis of Barrett esophagus requires endoscopic visualization of at least 1 cm of salmon-colored mucosa proximal to the gastroesophageal junction, along with biopsy confirmation of intestinal metaplasia and goblet cells. Endoscopy is typically performed in patients with chronic GERD symptoms, particularly if they have additional risk factors like male sex, older age, or a family history of Barrett esophagus or esophageal adenocarcinoma. Surveillance includes taking multiple biopsies from different quadrants to maximize diagnostic yield.

**Treatment**: The mainstay of treatment includes long-term proton pump inhibitors (PPIs) to reduce acid reflux and potentially prevent progression to dysplasia or cancer. If dysplasia or early cancer is present, endoscopic eradication therapies (EET) such as radiofrequency ablation, cryotherapy, and endoscopic mucosal resection may be used. Esophagectomy is considered in cases with high-grade dysplasia or invasive cancer. Endoscopic surveillance is crucial, with intervals based on the degree of dysplasia.

**Complications**: The primary complication of Barrett esophagus is the risk of progression to esophageal adenocarcinoma, which occurs more frequently with the presence of dysplasia. The progression is slow, and less than 5% of patients with Barrett esophagus will develop cancer. Other complications include reflux esophagitis, bleeding, or ulceration. Patients may also experience strictures or aspiration, though these are not more common than in non-Barrett esophagus cases. Regular surveillance and early intervention can significantly reduce the risk of malignancy.
"""
]

# Responses from the base model (provided earlier)
base_model_output = [
    """
Barrett's Esophagus (BE) is a condition where the lining of the 
esophagus, the tube that connects the mouth to the stomach, changes        
to resemble the lining of the intestine. Here's what you need to know      
about this potential precursor to esophageal cancer:

1. Causes: Barrett's Esophagus is primarily caused by long-term 
gastroesophageal reflux disease (GERD), where stomach acid frequently      
flows back into the esophagus.

2. Symptoms: Many people with Barrett's Esophagus don't exhibit any        
symptoms, but some may experience heartburn, difficulty swallowing,        
and regurgitation of food or sour liquid.

3. Diagnosis: A doctor can diagnose Barrett's Esophagus through an         
endoscopy, where a flexible tube with a camera is inserted into the        
esophagus to examine its lining. Biopsies may also be taken during         
the procedure.

4. Complications: The most significant complication of Barrett's 
Esophagus is the development of esophageal cancer, which occurs in
about 1% of patients per year. Regular screenings and careful
monitoring are essential for those with BE.

5. Treatment: Treatment options for Barrett's Esophagus include acid       
suppression therapy, eliminating GERD symptoms, and in some cases,         
endoscopic procedures to remove the affected tissue (ablation 
therapy). Surgery may also be an option in severe cases.

6. Prevention: Maintaining a healthy weight, avoiding alcohol and
tobacco use, and managing GERD with lifestyle changes or medication        
can help reduce the risk of developing Barrett's Esophagus. Regular        
screenings are recommended for those at higher risk.
    """
]

# Responses from the RAG-based model (custom model with the database)
rag_model_output = [
    """
### Barrett's Esophagus Overview:

Barrett's esophagus is a condition where the normal stratified squamous epithelium of the esophagus is replaced by specialized columnar epithelium. This change occurs due to prolonged acid injury from chronic gastroesophageal reflux disease (GERD).

   ### Presentation and Symptoms:

Barrett's esophagus may not cause any noticeable symptoms in the early stages. However, common signs of GERD, such as heartburn, acid regurgitation, and difficulty swallowing, can be present.

   ### Pathophysiology:

Barrett's metaplasia occurs due to chronic tissue injury in the esophagus caused by GERD. Risk factors include obesity, cigarette smoking, and a genetic predisposition.

   ### Diagnosis:

Diagnosis of Barrett's esophagus typically involves an endoscopic examination during evaluation for GERD or other indications.

   ### Treatment:

Treatment options for Barrett's esophagus include medications to reduce acid production, lifestyle changes such as weight loss and avoiding trigger foods, and in some cases, surgical interventions. 

   ### Complications:

If left untreated, Barrett's esophagus can increase the risk of developing esophageal adenocarcinoma over time [StatPearls]. It is essential to monitor patients with Barrett's esophagus closely and consider endoscopic surveillance for early detection and treatment.
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
