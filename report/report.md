# Aspect-Based Sentiment Analysis of Product and App Reviews
**Abstract**—This project presents a comparative study of various approaches to Aspect-Based Sentiment Analysis (ABSA) on the SemEval-2014 Task 4 datasets. We tackle two subtasks: Aspect Term Extraction (ATE) and Aspect Sentiment Classification (ASC). Our baseline models include a lexicon rule-based approach and TF-IDF classifiers, which are compared against classical machine learning models (CRFs) and modern Transformer architectures like BERT and DeBERTa. 

## I. Introduction
Aspect-Based Sentiment Analysis (ABSA) provides a more granular understanding of customer feedback than traditional sentence-level sentiment analysis. Instead of predicting a single polarity for a review, ABSA identifies the specific entities (aspects) discussed and the sentiment toward each. This project implements an end-to-end ABSA pipeline to process laptop and restaurant reviews.

## II. Related Work
ABSA has been heavily researched, especially following the SemEval-2014 Task 4 challenge [1]. Early approaches relied on lexicons and dependency parsing [2]. Later, Conditional Random Fields (CRFs) [3] and Support Vector Machines [4] became the standard for extraction and classification. Recently, Transformers such as BERT [5], RoBERTa [6], and DeBERTa [7] have achieved state-of-the-art results. Works like Sun et al. (2019) [8] reformulated ABSA as a sentence-pair classification task, significantly improving performance.

## III. Dataset
We utilized the SemEval-2014 Task 4 datasets for Laptops and Restaurants. The data consists of customer reviews annotated with aspect terms and polarities (positive, negative, neutral, conflict). The "conflict" label was dropped to frame the problem as a 3-class classification task. We performed a 90-10 split for training and validation. Exploratory Data Analysis revealed that the 'Restaurants' domain contained more aspects per sentence on average than the 'Laptops' domain.

## IV. Methodology
**Task A: Aspect Term Extraction (ATE)**
1. **Rule-Based**: Extracts noun phrases connected to opinion words via dependency relations.
2. **Classical ML**: A CRF model utilizing POS tags, casing, and word suffixes as features.
3. **Transformers**: Fine-tuned BERT using token classification with BIO tags.

**Task B: Aspect Sentiment Classification (ASC)**
1. **Baselines**: Logistic Regression and Linear SVM using TF-IDF of the sentence and a 10-word aspect window.
2. **Transformers**: Fine-tuned BERT using a sentence-pair formulation (`[CLS] text [SEP] aspect [SEP]`).

## V. Experiments and Results
The models were evaluated using Precision, Recall, and F1 for ATE, and Accuracy and Macro-F1 for ASC. The Transformer models heavily outperformed the baselines in both tasks, validating the strength of contextualized embeddings. Detailed metrics are available in the `results/tables/model_comparison.csv` file. End-to-end evaluation demonstrated a compounding effect of errors propagating from the extraction step to the classification step.

## VI. Error Analysis
A qualitative review of 50 misclassified examples revealed distinct error categories:
- **Complex sentence structures**: Sarcasm or long-distance dependencies often confused the simpler models.
- **Implicit aspects**: Missing explicit nouns made extraction difficult.
- **Negation scope**: "Not good" being classified as positive due to the presence of "good".

## VII. Limitations and Conclusion
While Transformers achieve high accuracy, they suffer from high computational costs. Future work could explore knowledge distillation or lighter architectures for edge-device deployment.

## References
[1] Pontiki, M., et al. (2014). SemEval-2014 Task 4: Aspect Based Sentiment Analysis.
[2] Hu, M., & Liu, B. (2004). Mining and summarizing customer reviews.
[3] Lafferty, J., et al. (2001). Conditional random fields: Probabilistic models for segmenting and labeling sequence data.
[4] Cortes, C., & Vapnik, V. (1995). Support-vector networks.
[5] Devlin, J., et al. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.
[6] Liu, Y., et al. (2019). RoBERTa: A Robustly Optimized BERT Pretraining Approach.
[7] He, P., et al. (2020). DeBERTa: Decoding-enhanced BERT with Disentangled Attention.
[8] Sun, C., et al. (2019). Utilizing BERT for Aspect-Based Sentiment Analysis via Constructing Auxiliary Sentence.
