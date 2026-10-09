# Aspect-Based Sentiment Analysis (ABSA) Mini-Project

## Slide 1: Title Slide
- **Title**: Aspect-Based Sentiment Analysis of Product and App Reviews
- **Subtitle**: Extracting Fine-Grained Insights from Customer Feedback
- **Speaker Notes**: Welcome everyone. Today I'll present our ABSA mini-project, aiming to move beyond traditional sentence-level sentiment analysis to extract specific entity aspects and their corresponding sentiments.

## Slide 2: Problem Statement
- **Point 1**: Customers write complex reviews containing multiple opinions (e.g., "Great food, bad service").
- **Point 2**: Standard sentiment analysis fails to capture these nuances.
- **Speaker Notes**: A review can be both positive and negative depending on the aspect. ABSA breaks this down into two tasks: finding the aspect term (like 'food') and determining its specific polarity.

## Slide 3: Motivation
- **Point 1**: Businesses need actionable insights.
- **Point 2**: Granular analysis drives product development and customer service improvements.
- **Speaker Notes**: By aggregating aspect-level sentiment, companies can pinpoint exactly what features they need to improve rather than just knowing people are generally unhappy.

## Slide 4: Data
- **Point 1**: SemEval-2014 Task 4 (Laptops and Restaurants).
- **Point 2**: BIO-tagged for extraction, and triplet formatted for classification.
- **Speaker Notes**: We used the gold-standard SemEval dataset. We dropped 'conflict' labels to formulate a clean 3-class problem: positive, negative, and neutral.

## Slide 5: Pipeline Overview
- **Step 1**: Preprocessing & BIO Tagging.
- **Step 2**: Aspect Term Extraction (ATE).
- **Step 3**: Aspect Sentiment Classification (ASC).
- **Speaker Notes**: Our pipeline is modular. We process the raw XMLs, feed them into extraction models, and pipe the predicted terms into classification models to get the final triplets.

## Slide 6: Models
- **ATE**: Rule-based (Dependency Parsing), CRF, and BERT Token Classification.
- **ASC**: TF-IDF + Logistic Regression/SVM, and BERT Sequence Classification.
- **Speaker Notes**: We compared traditional lightweight machine learning models against state-of-the-art transformer architectures.

## Slide 7: Results
- **Comparison**: Transformers significantly outperform classical baselines.
- **End-to-End**: ATE errors bottleneck overall pipeline performance.
- **Speaker Notes**: As expected, BERT provides the highest accuracy. However, when evaluated end-to-end, if the aspect boundary is missed by even one token, the entire triplet is penalized, showing the difficulty of the joint task.

## Slide 8: Error Analysis
- **Categories**: Implicit aspects, Sarcasm, Negation scope.
- **Example**: "Not the best battery, but it works."
- **Speaker Notes**: We sampled 50 errors and categorized them. A common issue was long-distance dependencies and complex negation scopes confounding the simpler models.

## Slide 9: Demo
- **Tab 1**: Interactive single-review analysis.
- **Tab 2**: Bulk dashboard visualization.
- **Speaker Notes**: We built a Streamlit app to showcase the models. It highlights aspects in real-time and visualizes aggregated sentiment distributions for business intelligence.

## Slide 10: Conclusion & Limitations
- **Conclusion**: Deep learning is necessary for nuanced ABSA.
- **Limitations**: High compute cost; difficulty with implicit aspects.
- **Speaker Notes**: In conclusion, while transformers are highly accurate, they are computationally heavy. Future work will explore smaller distilled models for faster inference.
