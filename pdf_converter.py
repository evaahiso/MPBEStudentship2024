"""
# Import library
from langchain_community.document_loaders import PyPDFLoader

# Create a document loader for rag_vs_fine_tuning.pdf
loader = PyPDFLoader('1 Microcontroller Programming (v1.2).pdf')

# Load the document
data = loader.load()
print(data[0])
print(len(data))
"""
import pandas as pd

df = pd.read_parquet("hf://datasets/microsoft/orca-math-word-problems-200k/data/train-00000-of-00001.parquet")

from cdqa.utils.converters import df2squad

json_data = df2squad(df=df, squad_version='v1.1', output_dir='.', filename='dataset-name')