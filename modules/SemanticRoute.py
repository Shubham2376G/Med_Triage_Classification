from semantic_router import Route
from semantic_router.encoders import FastEmbedEncoder
from semantic_router.layer import RouteLayer
import pandas as pd

def Routing(embedding_model="sentence-transformers/all-MiniLM-L6-v2",dataset="Classification_v1.json"):
    encoder = FastEmbedEncoder(model_name=embedding_model)

    # Load JSON file
    df = pd.read_json(f"Datasets/{dataset}")

    routine_carel = df[df['label'] == 0]['text'].tolist()
    scheduled_operationsl = df[df['label'] == 1]['text'].tolist()
    emergencyl = df[df['label'] == 2]['text'].tolist()
    urgent_carel = df[df['label'] == 3]['text'].tolist()



    routine_care=Route(
        name="routine_care",
        utterances=routine_carel,
        description="routine care",
    )
    urgent_care=Route(
        name="urgent_care",
        utterances=urgent_carel,
        description="urgent care"
    )
    emergency=Route(
        name="emergency",
        utterances=emergencyl,
        description="emergency",
    )
    scheduled_operations=Route(
        name="scheduled_operations",
        utterances=scheduled_operationsl,
        description="scheduled operations",
    )


    med_routes=[routine_care,urgent_care,emergency,scheduled_operations]
    med = RouteLayer(encoder=encoder, routes=med_routes)

    return med