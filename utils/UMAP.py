import pandas as pd
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
import seaborn as sns
import umap
import plotly.express as px


def umap_plot(dataset,interactive:bool=False):
    # dataset="Classification_v1.json"
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
    sentences=df["text"].to_list()
    labels=df["label"].to_list()
    # Sentences to embeddings
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = embedding_model.encode(sentences)


    #reduce dimensions to 2D using UMAP with adjusted n_neighbors
    umap_reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=5)
    reduced_embeddings = umap_reducer.fit_transform(embeddings)

    x1=[]
    y1=[]

    for i, sentence in enumerate(sentences):
        x1.append(reduced_embeddings[i][0])
        y1.append(reduced_embeddings[i][1])

    dict1={"sentence":sentences,"x":x1,"y":y1,"label":labels}
    df=pd.DataFrame(dict1)

    if interactive:
        plt.figure(figsize=(8, 6))
        fig = px.scatter(df,x="x", y="y", hover_name="sentence",color='label', title="Interactive Scatter Plot")
        fig.update_traces(marker=dict(size=12))
        fig.show()

    else:
        plt.figure(figsize=(8,6))
        sns.scatterplot(data=df, x='x', y='y', hue='label', palette='Set1')
        plt.title("Scatter Plot UMAP")
        plt.savefig("Visual_plots/UMAP_Visual.png", dpi=300, bbox_inches='tight')
        plt.show()