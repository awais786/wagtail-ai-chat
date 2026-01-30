"""
URL configuration for Wagtail RAG app.
"""
from django.urls import path

from . import views

app_name = 'wagtail_rag'

urlpatterns = [
    path('api/rag/chat/', views.rag_chat_api, name='rag_chat_api'),
    path('chatbox/', views.rag_chatbox_widget, name='rag_chatbox_widget'),
]

