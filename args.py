import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Med Classification")
    parser.add_argument('--embedding', type=str, help="select the embedding model",default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument('--dataset', type=str, help="dataset file path",default="Classification_v1.json")
    parser.add_argument('--semantic', action='store_true', help="Enable semantic model")
    parser.add_argument('--bert', action='store_true', help="Enable bert model")
    parser.add_argument('--tsne', action='store_true', help="plots tsne graph")
    parser.add_argument('--umap', action='store_true', help="plots umap graph")
    parser.add_argument('-i','--interactive', action='store_true', help="to get a interactive plot")

    return parser.parse_args()