import pandas as pd

from args import parse_args
from modules.SemanticRoute import Routing
from modules.ModernBert import modernbert_model
from utils.t_SNE import tsne_plot
from utils.UMAP import umap_plot
import gradio as gr

content=pd.read_excel("Examples.xlsx")
content=content["text"].to_list()
content=[[i] for i in content]

def semantics(query):
    embedding_model = args.embedding
    dataset = args.dataset
    semantic_model = Routing(embedding_model, dataset)
    return semantic_model(query).name

def bert_model(query):
    model=modernbert_model()
    return model(query)

def create_header():
    return """
    <h1 style="text-align: center; color: #4CAF50;">Medical Classification</h1>
    <p style="text-align: center; font-size: 18px; color: #555;">
        Classifies patient docs into 4 classes
    </p>

    """



args = parse_args()
if args.semantic:
    iface = gr.Interface(
        fn=semantics,  # Function to handle input
        inputs="text",  # Input type
        outputs="text",  # Output type
        examples=content,
        title="Semantic Routing",  # App title
        description=create_header()
    )
    # Launch the interface
    iface.launch()

elif args.bert:
    iface = gr.Interface(
        fn=bert_model,  # Function to handle input
        inputs="text",  # Input type
        outputs="text",  # Output type
        examples=content,
        title="ModernBert",  # App title
        description=create_header()
    )
    # Launch the interface
    iface.launch()

elif args.tsne:
    tsne_plot(args.dataset,args.interactive)

elif args.umap:
    umap_plot(args.dataset,args.interactive)





