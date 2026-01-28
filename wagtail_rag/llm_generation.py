"""
LLM Generation Module for RAG Chatbot.

This module handles all LLM (Large Language Model) generation functionality,
including prompt construction, chain execution, and answer generation.
"""
import logging

from django.conf import settings

logger = logging.getLogger(__name__)

# Try to import LangChain components
try:
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import RunnablePassthrough
    from langchain_core.output_parsers import StrOutputParser
    USE_LCEL = True
except ImportError:
    ChatPromptTemplate = None
    RunnablePassthrough = None
    StrOutputParser = None
    try:
        from langchain.prompts import PromptTemplate
        from langchain.chains import RetrievalQA
        USE_LCEL = False
    except ImportError:
        # Fallback - use simple pattern
        PromptTemplate = None
        RetrievalQA = None
        USE_LCEL = None


class LLMGenerator:
    """
    Handles LLM-based answer generation for RAG chatbot.
    
    This class encapsulates all LLM generation logic including:
    - Prompt template construction
    - QA chain setup (LCEL or legacy)
    - Answer generation with context
    """
    
    def __init__(self, llm, retriever=None):
        """
        Initialize the LLM generator.
        
        Args:
            llm: LangChain LLM instance
            retriever: Optional LangChain retriever (for LCEL chains)
        """
        self.llm = llm
        self.retriever = retriever
        
        # Create prompt template
        # NOTE: Default prompt is intentionally simple and avoids meta-comments like
        # "the context provided does not contain...". It asks the LLM to answer
        # directly from context, and if some detail is missing, simply not to mention it.
        prompt_template_str = getattr(
            settings,
            'WAGTAIL_RAG_PROMPT_TEMPLATE',
            """You are a helpful assistant for a bakery website.
Use ONLY the following context from the site to answer the question.
If the context does not mention some detail, simply do not talk about that detail.
Do not say things like "the context does not contain" or explain what is missing.

Context:
{context}

Question:
{question}

Answer:"""
        )
        
        # Create QA chain based on available LangChain version
        self.qa_chain = None
        if USE_LCEL:
            # Use LCEL pattern (LangChain 0.1+)
            prompt = ChatPromptTemplate.from_template(prompt_template_str)
            
            def format_docs(docs):
                return "\n\n".join(doc.page_content for doc in docs)
            
            if retriever:
                self.qa_chain = (
                    {
                        "context": retriever | format_docs,
                        "question": RunnablePassthrough()
                    }
                    | prompt
                    | llm
                    | StrOutputParser()
                )
            else:
                self.qa_chain = None
        elif USE_LCEL is False:
            # Use old RetrievalQA pattern
            prompt = PromptTemplate(
                template=prompt_template_str,
                input_variables=["context", "question"]
            )
            if retriever:
                self.qa_chain = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type="stuff",
                    retriever=retriever,
                    return_source_documents=True,
                    chain_type_kwargs={"prompt": prompt}
                )
            else:
                self.qa_chain = None
        else:
            # Simple fallback - just use LLM directly
            self.qa_chain = None
            self.prompt_template_str = prompt_template_str
    
    def generate_answer_with_llm(self, question, docs):
        """
        Generate an answer using the LLM with the retrieved context documents.
        
        Args:
            question: The user's question
            docs: List of Document objects retrieved via embedding search
            
        Returns:
            str: The LLM-generated answer
        """
        if self.qa_chain is None:
            # Simple fallback implementation - use retrieved docs directly
            context = "\n\n".join([doc.page_content for doc in docs])
            prompt_text = f"""Use the following context to answer the question.
If you don't know the answer, just say that you don't know.

Context: {context}

Question: {question}

Answer: """
            return self.llm(prompt_text)
        
        if USE_LCEL:
            # LCEL pattern - chain handles retrieval internally, but we pass docs for consistency
            # Note: The chain will retrieve again, but we use our retrieved docs for sources
            return self.qa_chain.invoke(question)
        else:
            # Old RetrievalQA pattern
            result = self.qa_chain({"query": question})
            return result['result']
    
    def generate_answer(self, question, docs=None):
        """
        Generate an answer using the LLM.
        
        This is the main entry point for LLM generation. It handles different
        LangChain versions and patterns.
        
        Args:
            question: The user's question
            docs: Optional list of Document objects (for fallback mode)
            
        Returns:
            str: The LLM-generated answer
        """
        if self.qa_chain is None:
            # Fallback mode - need docs
            if docs is None:
                raise ValueError("docs required when qa_chain is None")
            return self.generate_answer_with_llm(question, docs)
        
        if USE_LCEL:
            # LCEL pattern - chain handles retrieval internally
            return self.qa_chain.invoke(question)
        else:
            # Old RetrievalQA pattern
            result = self.qa_chain({"query": question})
            return result['result']
    
    def get_source_documents_from_chain(self, question):
        """
        Get source documents from the QA chain (for legacy RetrievalQA).
        
        Args:
            question: The user's question
            
        Returns:
            List of Document objects from the chain
        """
        if not USE_LCEL and self.qa_chain:
            result = self.qa_chain({"query": question})
            return result.get('source_documents', [])
        return []

