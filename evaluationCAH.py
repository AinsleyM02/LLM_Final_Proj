from bert_score import score
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Define the ground truth based on trusted sources (e.g., NIH, medical textbooks)


ground_truth = [
    """
- **Overview**:  
Congenital adrenal hyperplasia (CAH) refers to a group of autosomal recessive disorders caused by mutations in enzymes involved in the synthesis of corticosteroids in the adrenal glands. These mutations impair the production of cortisol, leading to a compensatory increase in adrenocorticotropic hormone (ACTH), causing adrenal hyperplasia. The condition can manifest in infants, children, or adults and may be associated with either a deficiency or excess of certain hormones depending on the specific enzymatic defect.

- **Presentation and Symptoms**:  
Symptoms of CAH vary based on the severity and specific form of the condition. Common signs include failure to thrive, hyperpigmentation, electrolyte imbalances (such as hyponatremia and hyperkalemia in salt-wasting forms), and adrenal crisis. In females, excessive androgen production can lead to virilization, including ambiguous genitalia and early puberty. In males, symptoms may include penile enlargement or, in some cases, under-virilization. Nonclassic CAH may present with milder symptoms, including early pubarche, hirsutism, and fertility issues in adulthood.

- **Pathophysiology**:  
CAH results from mutations in various genes involved in steroidogenesis. The most common form, 21-hydroxylase deficiency, leads to impaired cortisol and aldosterone synthesis, causing a buildup of precursor hormones and an increase in androgen production. The adrenal glands become enlarged due to the accumulation of steroid precursors, and ACTH levels rise in response to cortisol deficiency. In some forms of CAH, mineralocorticoid precursors accumulate, leading to hypertension, hypokalemia, and metabolic alkalosis.

- **Diagnosis**:  
Diagnosis is often made through newborn screening that measures 17-hydroxyprogesterone (17-OHP) levels. Elevated 17-OHP levels indicate the presence of 21-OH deficiency. Additional diagnostic tests, such as cosyntropin stimulation tests and measurement of other hormones (e.g., cortisol, progesterone, and dehydroepiandrosterone), can help confirm the specific type of CAH. Genetic testing may be used to identify specific enzyme deficiencies.

- **Treatment**:  
Treatment typically involves hormone replacement therapy to address cortisol deficiency. Hydrocortisone is commonly used to manage glucocorticoid insufficiency, and fludrocortisone is used for mineralocorticoid replacement in salt-wasting forms. In cases of excessive androgen production, treatments to control androgen levels may include oral contraceptives or anti-androgens. Surgery may be required for individuals with ambiguous genitalia or other anatomical concerns. Lifelong monitoring and stress-dosing of steroids during illness or physical stress are essential to prevent adrenal crisis.

- **Complications**:  
Complications include adrenal crisis, which can be life-threatening without prompt treatment, and long-term issues such as growth abnormalities, infertility, and progressive virilization. CAH patients may have a higher risk of metabolic disorders, such as obesity, insulin resistance, and cardiovascular disease, particularly due to prolonged glucocorticoid treatment. Females may experience challenges related to sexual function and fertility, and males may develop testicular adrenal rest tumors (TARTs), which can impair fertility. Management of these complications is crucial to improving the quality of life for affected individuals.
"""
]

# Responses from the base model (provided earlier)
base_model_output = [
    """
Congenital Adrenal Hyperplasia (CAH) is a group of inherited disorders caused by mutations in the genes that produce enzymes necessary for      
the production of cortisol and aldosterone, hormones produced by the adrenal glands. Here's what you need to know about this condition:

1. Causes: CAH results from a deficiency or absence of certain enzymes in the adrenal gland pathway that produces cortisol and aldosterone,      
due to genetic mutations. The most common form of CAH is 21-hydroxylase deficiency.

2. Symptoms: In females, symptoms may include ambiguous genitalia at birth, delayed onset of puberty, irregular menstrual periods, and 
infertility. Males with CAH can also experience delayed puberty and decreased fertility, as well as excessive growth of the adrenal glands       
(hyperplasia) and enlargement of other male organs.

3. Diagnosis: CAH is diagnosed through a series of tests, including blood tests to measure hormone levels and genetic testing to identify        
specific mutations in the genes responsible for the condition.

4. Complications: Complications of untreated CAH can include life-threatening salt imbalances, poor growth, and early puberty in females.        
In males, complications may include low testosterone levels, decreased fertility, and increased risk of developing testicular cancer.

5. Treatment: Treatment for CAH involves replacement therapy with synthetic hormones to compensate for the deficiencies caused by the
genetic mutations. Surgery may also be necessary in some cases to correct ambiguous genitalia in females or remove enlarged adrenal glands.

6. Prevention: Since CAH is inherited, there is no way to prevent it from occurring. However, early diagnosis and treatment can
significantly improve the quality of life for individuals with CAH and reduce the risk of complications. Genetic testing is available for        
at-risk families to determine if they are carriers of the mutated genes responsible for CAH.
    """
]

# Responses from the RAG-based model (custom model with the database)
rag_model_output = [
    """
**Overview**: Congenital Adrenal Hyperplasia (CAH) is a genetic disorder that affects the adrenal glands' ability to produce certain hormones, particularly cortisol and aldosterone. This condition can lead to various health problems due to the accumulation of steroid precursors in the body.

**Presentation and Symptoms**: The symptoms of CAH can vary widely depending on the specific enzyme deficiency. In females with 21-hydroxylase deficiency (the most common form), there might be signs of virilization such as clitoromegaly, fusion of the labia majora, and other masculinizing features in a newborn female. Older individuals may experience symptoms like fatigue, low blood pressure, weak muscles, irregular periods, and infertility.

**Pathophysiology**: In CAH, the adrenal glands fail to produce adequate amounts of cortisol and aldosterone due to a deficiency in an enzyme involved in hormone synthesis. As a result, the body produces an excess of steroid precursors, particularly androgens, which can cause virilization and other symptoms.

**Diagnosis**: CAH is typically diagnosed through genetic testing, blood tests to measure hormone levels, and imaging studies to assess adrenal gland size. Newborn screening programs are also in place to identify affected individuals early on.

**Treatment**: Treatment for CAH involves replacing the missing hormones (cortisol and aldosterone) through medication. Lifelong hormonal replacement is necessary to manage symptoms and prevent complications . Additionally, salt restriction and fluid intake monitoring may be recommended for those with aldosterone deficiency.

**Complications**: If left untreated or inadequately managed, CAH can lead to a variety of complications such as high blood pressure, low blood sugar levels, dehydration, and growth retardation. In severe cases, it can also result in adrenal crisis, which is a life-threatening condition requiring immediate medical attention.
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
