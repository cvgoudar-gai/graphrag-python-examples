import os
from dotenv import load_dotenv
import neo4j
from neo4j_graphrag.llm import OpenAILLM
from neo4j_graphrag.embeddings.openai import OpenAIEmbeddings
from neo4j_graphrag.retrievers import VectorRetriever, VectorCypherRetriever
from neo4j_graphrag.generation import RagTemplate
from neo4j_graphrag.generation.graphrag import GraphRAG
from neo4j_graphrag.indexes import create_vector_index

# Load environment variables from the root directory
load_dotenv('../.env', override=True)

class GraphRAGApp:
    def __init__(self):
        self.setup_neo4j_connection()
        self.setup_models()
        self.setup_retrievers()
        self.setup_graphrag_pipelines()
    
    def setup_neo4j_connection(self):
        """Setup Neo4j connection"""
        self.NEO4J_URI = os.getenv('NEO4J_URI')
        self.NEO4J_USERNAME = os.getenv('NEO4J_USERNAME')
        self.NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD')
        self.NEO4J_DATABASE = os.getenv('NEO4J_DATABASE')
        
        if not all([self.NEO4J_URI, self.NEO4J_USERNAME, self.NEO4J_PASSWORD]):
            raise ValueError("Neo4j credentials not found in environment variables")
        
        self.driver = neo4j.GraphDatabase.driver(
            self.NEO4J_URI, 
            auth=(self.NEO4J_USERNAME, self.NEO4J_PASSWORD), 
            database=self.NEO4J_DATABASE
        )
    
    def setup_models(self):
        """Setup LLM and embedding models"""
        self.llm = OpenAILLM(
            model_name="gpt-4.1-mini",
            model_params={
                "response_format": {"type": "json_object"},
                "temperature": 0
            }
        )
        
        self.embedder = OpenAIEmbeddings()
        
        # Create vector index if it doesn't exist
        try:
            create_vector_index(
                self.driver, 
                name="text_embeddings", 
                label="Chunk",
                embedding_property="embedding", 
                dimensions=1536, 
                similarity_fn="cosine"
            )
        except Exception as e:
            print(f"Vector index creation error (may already exist): {e}")
    
    def setup_retrievers(self):
        """Setup vector and vector-cypher retrievers"""
        self.vector_retriever = VectorRetriever(
            self.driver,
            index_name="text_embeddings",
            embedder=self.embedder,
            return_properties=["text"],
        )
        
        self.vc_retriever = VectorCypherRetriever(
            self.driver,
            index_name="text_embeddings",
            embedder=self.embedder,
            retrieval_query="""
//1) Go out 2-3 hops in the entity graph and get relationships
WITH node AS chunk
MATCH (chunk)<-[:FROM_CHUNK]-()-[relList:!FROM_CHUNK]-{1,2}()
UNWIND relList AS rel

//2) collect relationships and text chunks
WITH collect(DISTINCT chunk) AS chunks, 
  collect(DISTINCT rel) AS rels

//3) format and return context
RETURN '=== text ===\n' + apoc.text.join([c in chunks | c.text], '\n---\n') + '\n\n=== kg_rels ===\n' +
  apoc.text.join([r in rels | startNode(r).name + ' - ' + type(r) + '(' + coalesce(r.details, '') + ')' +  ' -> ' + endNode(r).name ], '\n---\n') AS info
"""
        )
    
    def setup_graphrag_pipelines(self):
        """Setup GraphRAG pipelines with custom prompt template"""
        rag_template = RagTemplate(
            template='''Answer the Question using the following Context. Only respond with information mentioned in the Context. Do not inject any speculative information not mentioned. 

# Question:
{query_text}
 
# Context:
{context}

# Answer:
''', 
            expected_inputs=['query_text', 'context']
        )
        
        generation_llm = OpenAILLM(
            model_name="gpt-4.1-mini",  
            model_params={"temperature": 0.0}
        )
        
        self.v_rag = GraphRAG(
            llm=generation_llm, 
            retriever=self.vector_retriever, 
            prompt_template=rag_template
        )
        
        self.vc_rag = GraphRAG(
            llm=generation_llm, 
            retriever=self.vc_retriever, 
            prompt_template=rag_template
        )
    
    def query_graphrag(self, query_text, top_k=5):
        """Query both GraphRAG pipelines and return results side by side"""
        try:
            # Get results from both retrievers
            v_result = self.v_rag.search(
                query_text, 
                retriever_config={'top_k': top_k}, 
                return_context=True
            )
            
            vc_result = self.vc_rag.search(
                query_text, 
                retriever_config={'top_k': top_k}, 
                return_context=True
            )
            
            # Format the results
            vector_response = f"**Vector Retriever Response:**\n\n{v_result.answer}"
            vc_response = f"**Vector + Cypher Retriever Response:**\n\n{vc_result.answer}"
            
            return vector_response, vc_response
            
        except Exception as e:
            error_msg = f"Error processing query: {str(e)}"
            return error_msg, error_msg
    
    def close_connection(self):
        """Close Neo4j connection"""
        if hasattr(self, 'driver'):
            self.driver.close() 