import os
import matplotlib.pyplot as plt
import numpy as np
from lime.lime_text import LimeTextExplainer
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def explain_predictions_lime(texts, predict_proba_fn, class_names, output_dir, num_samples=10):
    """
    predict_proba_fn: A function that takes a list of strings and returns a matrix of shape (len(texts), num_classes)
                      with the probability of each class.
    """
    os.makedirs(output_dir, exist_ok=True)
    explainer = LimeTextExplainer(class_names=class_names)
    
    samples = texts[:num_samples]
    
    html_output = "<html><body><h1>LIME Explanations</h1>"
    
    for i, text in enumerate(samples):
        exp = explainer.explain_instance(text, predict_proba_fn, num_features=6, num_samples=100)
        
        # Save figure
        fig = exp.as_pyplot_figure()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'lime_exp_{i}.png'))
        plt.close(fig)
        
        # Append to HTML
        html_output += f"<h2>Example {i+1}</h2>"
        html_output += f"<p><b>Text:</b> {text}</p>"
        html_output += exp.as_html()
        html_output += "<hr>"
        
    html_output += "</body></html>"
    
    with open(os.path.join(output_dir, "lime_explanations.html"), 'w', encoding='utf-8') as f:
        f.write(html_output)
        
    logging.info(f"Saved LIME explanations to {output_dir}")

def generate_word_cloud(texts, output_path, title="Word Cloud"):
    from wordcloud import WordCloud
    text_combined = " ".join(texts)
    if not text_combined.strip():
        return
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text_combined)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(title)
    plt.savefig(output_path)
    plt.close()
    logging.info(f"Saved WordCloud to {output_path}")

if __name__ == "__main__":
    pass
