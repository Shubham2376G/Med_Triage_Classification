from sentence_transformers import SentenceTransformer
from sklearn.manifold import TSNE
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px


def tsne_plot(dataset,interactive: bool = False):
    # dataset = "Classification_v1.json"
    df = pd.read_json(f"Datasets/{dataset}")

    # Mapping dictionary
    label_mapping = {
        0: 'routine care',
        1: 'scheduled surgery/operation',
        2: 'emergency',
        3: 'urgent care'
    }
    # Map numeric labels to class names
    df['label'] = df['label'].map(label_mapping)

    # Sentence data as a list
    sentences = df["text"].to_list()
    labels = df["label"].to_list()

    # Sentences to embeddings
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    sentence_embeddings = embedding_model.encode(sentences)

    # Reduce embedding dimensionality using t-SNE for visualization
    viz_embeddings = TSNE(n_components=2, perplexity=50, max_iter=8000, random_state=42).fit_transform(
        sentence_embeddings)

    dict1 = {"sentence": sentences, "x": viz_embeddings[:, 0], "y": viz_embeddings[:, 1], "label": labels}
    df = pd.DataFrame(dict1)

    if interactive:
        plt.figure(figsize=(8, 6))
        fig = px.scatter(df, x="x", y="y", hover_name="sentence", color='label', title="Interactive Scatter Plot")
        fig.update_traces(marker=dict(size=12))
        fig.show()


    else:
        plt.figure(figsize=(8, 6))
        sns.scatterplot(data=df, x='x', y='y', hue='label', palette='Set1')
        plt.title("Scatter Plot TSNE")
        plt.savefig("Visual_plots/TSNE_Visual.png", dpi=300, bbox_inches='tight')
        plt.show()