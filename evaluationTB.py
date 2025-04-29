from bert_score import score
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Define the ground truth based on trusted sources (e.g., NIH, medical textbooks)


ground_truth = [
    """

Overview
Tuberculosis (TB) is an infectious disease caused primarily by Mycobacterium tuberculosis (Mtb), responsible for more human deaths throughout history than any other infectious disease. TB remains a major global health challenge despite being preventable and curable. Diagnosis, treatment, and prevention are complicated by the slow growth of the organism, emerging drug resistance, and socioeconomic factors such as poverty and overcrowding.
Presentation and Symptoms
TB symptoms are often nonspecific and can range from asymptomatic to severe illness. Common signs include persistent cough, fever, weight loss, night sweats, and malaise. In pulmonary TB, lung exam findings may range from normal to areas of consolidation or cavities. Extrapulmonary TB can involve any organ, with corresponding symptoms. In HIV-infected individuals, TB often presents atypically, particularly when CD4+ counts are low.
Pathophysiology
TB is typically spread via airborne droplets. Inhaled bacilli can be killed, establish latent infection, or cause active disease. The bacteria survive within alveolar macrophages, leading to granuloma formation in the lungs. Latency involves immune containment, but bacilli can reactivate, especially when host immunity declines. Granulomas may necrotize and cavitate, facilitating transmission. Dissemination to extrapulmonary sites (e.g., pleura, meninges, bones) is common. HIV coinfection significantly alters the immune response and increases the risk of dissemination and mortality.
Diagnosis
Diagnosis depends on context: latent, active pulmonary, or extrapulmonary TB. Key tools include:

Screening for Latent TB: Mantoux tuberculin skin test (TST) and interferon-gamma release assays (IGRAs).

Diagnosis of Active TB: Chest x-ray, CT, microbiological cultures, acid-fast smears, and nucleic acid amplification tests (NAATs) like Xpert MTB/RIF. Culture remains the gold standard despite slow growth. ADA testing can assist with extrapulmonary TB (pleural, meningitis).

Drug Resistance Testing: Rapid NAATs and line probe assays are crucial for detecting drug-resistant TB strains.
Treatment
Latent TB: Shortened regimens such as 3 months of weekly isoniazid and rifapentine (3HP) are recommended to improve adherence and minimize toxicity.

Active TB: Standard regimens involve two phases—an intensive phase with 4 drugs (isoniazid, rifampin, pyrazinamide, ethambutol) followed by a continuation phase. A new 4-month regimen using rifapentine and moxifloxacin has been endorsed for certain populations.

Drug-Resistant TB: Requires longer treatment (18–21 months) with newer or repurposed drugs like bedaquiline, linezolid, and delamanid.

Vaccination: BCG vaccine protects infants and children from severe TB forms but offers limited adult protection and complicates TST interpretation.
Complications
Complications include severe organ damage (e.g., lung fibrosis, meningitis, osteomyelitis), septic shock, and infertility. Anti-TB therapy itself can cause significant side effects such as hepatitis, neuropathy, ocular toxicity, and drug interactions, particularly in TB-HIV coinfection. Socioeconomic barriers, inadequate treatment adherence, and emergence of drug-resistant strains exacerbate global TB control challenges.
"""
]

# Responses from the base model (provided earlier)
base_model_output = [
    """
Tuberculosis (TB) is an infectious disease caused by the bacterium        
Mycobacterium tuberculosis. It primarily affects the lungs but can
also affect other parts of the body. Here's what you need to know
about this ancient and persistent global health threat:

1. Transmission: Tuberculosis is transmitted through the air when
infected individuals cough, sneeze, or breathe near others. Prolonged      
exposure to someone with active TB increases the risk of infection.

2. Symptoms: Common symptoms include a persistent cough lasting more       
than three weeks, chest pain, coughing up blood, fever, night sweats,      
and weight loss. However, many people with latent TB (infection
without symptoms) may not exhibit any signs or symptoms.

3. Diagnosis: A doctor can diagnose tuberculosis through a series of       
tests, including a chest X-ray, skin test, or blood test.

4. Complications: Untreated or poorly managed TB can lead to severe        
lung damage, respiratory failure, and even death. It can also 
increase the risk of developing other opportunistic infections.

5. Treatment: Tuberculosis is treatable with a course of antibiotics,      
typically for 6 to 9 months. Directly observed therapy (DOT) is often      
recommended to ensure that patients take their medication correctly.

6. Prevention: Vaccination against TB (BCG vaccine) and maintaining        
good ventilation in crowded places can help prevent the spread of the      
disease. If you have been in close contact with someone with active        
TB, your healthcare provider may recommend a tuberculin skin test or       
blood test to determine if you've been infected
    """
]

# Responses from the RAG-based model (custom model with the database)
rag_model_output = [
    """
    **Overview**: Tuberculosis (TB) is a bacterial infection primarily affecting the lungs, although it can spread to other parts of the body. It is caused by Mycobacterium tuberculosis and is transmitted through the air when an infected person coughs or sneezes.

   **Presentation and Symptoms**: Common symptoms of TB include persistent coughing for more than three weeks, chest pain, coughing up blood, fever, night sweats, fatigue, and weight loss. In some cases, there may be no symptoms at all.

   **Pathophysiology**: When an individual inhales M. tuberculosis, it is ingested by macrophages but not destroyed. The bacteria survive and multiply within the macrophages, causing inflammation and forming a lesion called a tubercle. If left untreated, TB can spread to other parts of the body.

   **Diagnosis**: Diagnosing TB typically involves a physical examination, chest X-ray, and microbiological tests such as sputum smear or culture. A positive tuberculin skin test or interferon-gamma release assay can also be used to diagnose latent TB infection.

   **Treatment**: Treatment for TB involves a combination of antibiotics, usually isoniazid and rifampin, taken for six to nine months. Directly observed therapy (DOT), where medications are administered under supervision, may be recommended to ensure adherence.

   **Complications**: Complications of TB can include lung damage, respiratory failure, and the spread of infection to other parts of the body. In AIDS patients, the disease progression is generally more severe, potentially leading to chronic pneumonitis, tuberculous osteomyelitis, or tuberculous meningitis.
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
