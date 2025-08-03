import gradio as gr
from graphrag_core import GraphRAGApp

def create_gradio_interface():
    """Create and return the Gradio interface"""
    app = GraphRAGApp()
    
    # Custom CSS for ChatGPT-like styling
    custom_css = """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    .gradio-container {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif !important;
        font-weight: 400;
        line-height: 1.6;
    }
    
    .gradio-container h1, .gradio-container h2, .gradio-container h3 {
        font-weight: 600;
        color: #1a1a1a;
    }
    
    .gradio-container p, .gradio-container div {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif !important;
    }
    
    .gradio-container input, .gradio-container textarea, .gradio-container select {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif !important;
        font-size: 14px;
    }
    
    .gradio-container button {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif !important;
        font-weight: 500;
    }
    
    .gradio-container .markdown {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif !important;
        line-height: 1.6;
    }
    </style>
    """
    
    with gr.Blocks(title="GraphRAG Lupus Research Assistant", theme=gr.themes.Soft(), css=custom_css) as interface:
        gr.Markdown("# 🧬 GraphRAG Q&A Assistant")
        
        with gr.Row():
            # Main content area (3/4 width)
            with gr.Column(scale=3):
                # Query section
                query_input = gr.Textbox(
                    label="Your Question",
                    placeholder="e.g., How is precision medicine applied to Lupus?",
                    lines=3
                )
                
                submit_btn = gr.Button("Submit Query", variant="primary")
                
                # Results section
                with gr.Row():
                    with gr.Column(scale=1):
                        vector_output = gr.Markdown(
                            label="Vector Retriever Response",
                            value="**Vector Retriever Response:**\n\n*Submit a query to see results here*"
                        )
                    
                    with gr.Column(scale=1):
                        vc_output = gr.Markdown(
                            label="Vector + Cypher Retriever Response", 
                            value="**Vector + Cypher Retriever Response:**\n\n*Submit a query to see results here*"
                        )
            
            # Sidebar (1/4 width)
            with gr.Column(scale=1):
                gr.Markdown("### 📚 About This App")
                gr.Markdown("""
                This application demonstrates GraphRAG with two different retrieval strategies:
                
                - **Vector Retriever**: Uses vector similarity search on text chunks
                - **Vector + Cypher Retriever**: Combines vector search with graph traversal
                
                Ask questions about Lupus research and see how different retrieval methods provide different insights!
                """)
                
                gr.Markdown("### ⚙️ Settings")
                top_k_slider = gr.Slider(
                    minimum=1,
                    maximum=10,
                    value=5,
                    step=1,
                    label="Number of retrieved chunks (top_k)"
                )
                
                gr.Markdown("### 💡 Example Questions")
                gr.Markdown("""
                - How is precision medicine applied to Lupus?
                - Can you summarize systemic lupus erythematosus (SLE)?
                - What are the common biomarkers for Lupus?
                - What treatments are available for Lupus patients?
                """)
        
        def process_query(query, top_k):
            if not query.strip():
                return "Please enter a question.", "Please enter a question."
            
            vector_resp, vc_resp = app.query_graphrag(query, top_k)
            return vector_resp, vc_resp
        
        submit_btn.click(
            fn=process_query,
            inputs=[query_input, top_k_slider],
            outputs=[vector_output, vc_output]
        )
        
        # Handle Enter key in textbox
        query_input.submit(
            fn=process_query,
            inputs=[query_input, top_k_slider],
            outputs=[vector_output, vc_output]
        )
    
    return interface, app

if __name__ == "__main__":
    interface, app = create_gradio_interface()
    
    try:
        interface.launch(
            server_name="127.0.0.1",
            server_port=7880,
            share=False,
            show_error=True
        )
    finally:
        app.close_connection() 