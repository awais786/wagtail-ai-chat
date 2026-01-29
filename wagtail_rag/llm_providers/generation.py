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
        self.prompt_template_str = self._get_prompt_template()
        self.qa_chain = self._create_qa_chain()

    def _get_prompt_template(self):
        """Get the prompt template from settings or use default."""
        return getattr(
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

    def _create_qa_chain(self):
        """Create QA chain based on available LangChain version."""
        if not self.retriever:
            return None

        if USE_LCEL:
            return self._create_lcel_chain()
        elif USE_LCEL is False:
            return self._create_legacy_chain()

        return None

    def _create_lcel_chain(self):
        """Create LCEL-based chain (LangChain 0.1+)."""
        prompt = ChatPromptTemplate.from_template(self.prompt_template_str)

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        return (
            {
                "context": self.retriever | format_docs,
                "question": RunnablePassthrough()
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )

    def _create_legacy_chain(self):
        """Create legacy RetrievalQA chain."""
        prompt = PromptTemplate(
            template=self.prompt_template_str,
            input_variables=["context", "question"]
        )
        return RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt}
        )

    def _format_simple_prompt(self, question, docs):
        """Format a simple prompt when no chain is available."""
        context = "\n\n".join(doc.page_content for doc in docs)
        return self.prompt_template_str.format(context=context, question=question)

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
            # Simple fallback - use LLM directly with formatted prompt
            prompt_text = self._format_simple_prompt(question, docs)
            return self.llm(prompt_text)
        
        if USE_LCEL:
            # LCEL chain handles retrieval internally
            return self.qa_chain.invoke(question)

        # Legacy RetrievalQA pattern
        result = self.qa_chain({"query": question})
        return result['result']

    def generate_answer(self, question, docs=None):
        """
        Generate an answer using the LLM.
        
        This is the main entry point for LLM generation. It handles different
        LangChain versions and patterns.
        
        Args:
            question: The user's question
            docs: Optional list of Document objects (required for fallback mode)

        Returns:
            str: The LLM-generated answer
        """
        if self.qa_chain is None:
            if docs is None:
                raise ValueError("docs required when qa_chain is None")
            return self.generate_answer_with_llm(question, docs)

        # If docs provided, use them directly
        if docs:
            return self.generate_answer_with_llm(question, docs)

        # Use chain
        if USE_LCEL:
            return self.qa_chain.invoke(question)

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

