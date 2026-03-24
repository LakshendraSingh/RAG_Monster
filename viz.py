import chromadb
import streamlit as st
import pandas as pd
import plotly.express as px
import os
from dotenv import load_dotenv

load_dotenv()

CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma_db")

def main():
    st.set_page_config(page_title="Chroma DB Visualizer", layout="wide")
    st.title("Chroma Vector DB Explorer")

    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collections = client.list_collections()
    
    if not collections:
        st.warning("No collections found in the database.")
        return

    col_names = [c.name for c in collections]
    selected_col = st.sidebar.selectbox("Select Collection", col_names)

    collection = client.get_collection(selected_col)
    
    # Check if empty first
    if collection.count() == 0:
        st.info(f"Collection `{selected_col}` is currently empty.")
        return

    try:
        data = collection.get(include=["documents", "metadatas", "embeddings"])
    except Exception as e:
        st.error(f"Error accessing collection data: {e}")
        st.warning("This typically means the HNSW index on disk is missing or corrupted.")
        st.write("Suggested fixes:")
        st.write("1. Run `python cli.py index <file>` to ensure at least one document is indexed.")
        st.write("2. If it persists, run `python cli.py clear-db` and try again.")
        return

    st.header(f"Collection: `{selected_col}`")
    st.metric("Total Chunks", len(data["ids"]))

    if data["ids"]:
        df = pd.DataFrame({
            "ID": data["ids"],
            "Document": data["documents"],
            "Metadata": [str(m) for m in data["metadatas"]],
            "Vector Preview": [str(v[:5]) + "..." for v in data["embeddings"]] if data["embeddings"] is not None else ["N/A"] * len(data["ids"])
        })

        st.subheader("Data Overview")
        st.dataframe(df, use_container_width=True)

        # Dimensionality Reduction for Visualization
        if data["embeddings"] is not None and len(data["embeddings"]) > 1:
            st.subheader("Semantic Mapping")
            
            viz_mode = st.radio("Visualization Mode", ["2D (Fast)", "3D (Detailed)"], horizontal=True)
            
            from sklearn.decomposition import PCA
            import numpy as np
            
            embeddings = np.array(data["embeddings"])
            n_dims = 3 if viz_mode == "3D (Detailed)" and len(data["embeddings"]) > 2 else 2
            
            pca = PCA(n_components=n_dims)
            projections = pca.fit_transform(embeddings)
            
            if viz_mode == "3D (Detailed)" and n_dims == 3:
                proj_df = pd.DataFrame(projections, columns=["x", "y", "z"])
                proj_df["id"] = data["ids"]
                proj_df["text_preview"] = [d[:50] + "..." for d in data["documents"]]
                
                fig = px.scatter_3d(proj_df, x="x", y="y", z="z", hover_data=["id", "text_preview"], 
                                     title="3D Semantic Mapping (PCA)")
            else:
                proj_df = pd.DataFrame(projections[:, :2], columns=["x", "y"])
                proj_df["id"] = data["ids"]
                proj_df["text_preview"] = [d[:50] + "..." for d in data["documents"]]
                
                fig = px.scatter(proj_df, x="x", y="y", hover_data=["id", "text_preview"], 
                                 title="2D Semantic Mapping (PCA)")
            
            st.plotly_chart(fig, use_container_width=True)
        elif len(data["embeddings"]) == 1:
            st.info("Semantic mapping (PCA) requires at least 2 documents.")
    else:
        st.info("This collection is empty.")

if __name__ == "__main__":
    main()
