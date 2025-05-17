from django.urls import path
from django.views.generic import TemplateView
from django.contrib.auth import views as auth_views
from django.conf import settings
from django.conf.urls.static import static
from . import views

app_name = 'docs'

urlpatterns = [
    path('', TemplateView.as_view(template_name='docs/home.html'), name='home'),
    # Authentication
    path('login/', auth_views.LoginView.as_view(template_name='docs/login.html'), name='login'),
    path('logout/', auth_views.LogoutView.as_view(template_name='docs/logout.html'), name='logout'),
    path('signup/', views.signup_view, name='signup'),
    
    # Core Functionality
    path('dashboard/', views.dashboard, name='dashboard'),
    path('upload/', views.upload_project, name='upload'),
    
    # Document Handling
    path('project/<int:project_id>/preview/', views.preview_document, name='preview'),
    path('project/<int:project_id>/<str:format_type>/', views.document_view, name='document'),
    path('document/<int:doc_id>/download/', views.download_document, name='download_document'),
    
    # Document download
    path('project/<int:project_id>/download/markdown/', views.download_markdown, name='download_markdown'),
    path('project/<int:project_id>/download/html/', views.download_html, name='download_html'),
    path('project/<int:project_id>/download/pdf/', views.download_pdf, name='download_pdf'),
    path('project/<int:project_id>/download/<str:doc_type>/', views.download_doc, name='download_document'),
    path('delete-file/', views.delete_file, name='delete_file'),
    path('download/<path:path>/', views.download_file, name='download_file'),
    path('chat/', views.chat_view, name='chat'),
    path('api/chat/', views.chat_endpoint, name='chat_endpoint'),
    path('api/upload-document/', views.upload_document, name='upload_document'),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)