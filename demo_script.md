# Demo Script (3 Minutes)

**0:00 - 0:30**: "Hello, today I am presenting our Aspect-Based Sentiment Analysis pipeline. We tackle the problem of granular review analysis. Instead of saying a whole review is positive, our system finds what specifically the customer liked or disliked."

**0:30 - 1:15**: "Here is our Streamlit app. Let's look at Tab 1. I'll paste a complex review: 'The battery life is amazing but the screen is terrible.' I click analyze, and as you can see, it correctly highlights 'battery life' in green (positive) and 'screen' in red (negative). This is driven by our fine-tuned transformer models running on the backend."

**1:15 - 2:00**: "Now, moving to Tab 2. Imagine a business owner has a CSV of thousands of reviews. We upload this file here. The dashboard instantly generates insights. You can see the Aspect Frequency chart—showing 'food' and 'service' are the most talked-about. Below it, the sentiment split reveals that while 'food' is mostly positive, 'service' has a high negative ratio, immediately highlighting a pain point for the business."

**2:00 - 2:45**: "Under the hood, we experimented with Rule-based, CRF, and BERT models. BERT achieved the highest F1 score but we also explored classical ML which is much faster for CPU inference. Our error analysis showed that implicit aspects—where a noun isn't explicitly mentioned—remain the toughest challenge."

**2:45 - 3:00**: "This concludes the demo. Our pipeline is fully modular, open-sourced, and provides actionable insights for any review dataset. Thank you."

---

# Top 10 Viva Questions and Answers

1. **Q: What is the difference between standard sentiment analysis and ABSA?**
   *A: Standard sentiment assigns one polarity to an entire text. ABSA identifies specific entities (aspects) in the text and assigns a polarity to each individual aspect.*

2. **Q: Why did you drop the 'conflict' polarity from the dataset?**
   *A: 'Conflict' instances are rare and often represent noisy or ambiguous annotations. Dropping them simplifies the task to a standard 3-class classification problem without losing much practical value.*

3. **Q: How did you format the input for the Transformer ASC model?**
   *A: We used a sentence-pair formulation: `[CLS] text [SEP] aspect [SEP]`. This allows the model's self-attention mechanism to naturally model the relationship between the sentence context and the specific aspect.*

4. **Q: Why use BIO tagging for Aspect Term Extraction?**
   *A: Aspects can span multiple words (e.g., "battery life"). BIO tagging (Begin, Inside, Outside) allows token-level classifiers to correctly identify multi-word aspect boundaries.*

5. **Q: What features were used in your CRF baseline?**
   *A: We used the token itself, its POS tag, casing features, suffixes, and a window of neighboring tokens (context).*

6. **Q: Explain the end-to-end evaluation metric.**
   *A: In end-to-end evaluation, a prediction is only counted as a True Positive if both the exact boundary of the aspect term is correctly extracted AND its predicted polarity is correct.*

7. **Q: What was a common error made by the models?**
   *A: Negation scope issues. If a review says "The food was not good at all", simpler models see "good" and predict positive, missing the negation.*

8. **Q: Why did you use class weights in the ASC training?**
   *A: The dataset is imbalanced (often skewed towards positive reviews). Class weights heavily penalize misclassifications of minority classes (like neutral or negative), improving the Macro-F1 score.*

9. **Q: How does LIME explain the model's predictions?**
   *A: LIME generates local perturbations of the input text and observes how the model's probability changes, building a simple linear surrogate model to highlight which words contributed most to the prediction.*

10. **Q: If you had more time, how would you improve the ATE model?**
    *A: I would explore joint models that extract aspects and sentiments simultaneously, preventing the cascading errors from the ATE step from degrading the ASC performance.*
