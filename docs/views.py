from importlib.resources import contents
from django.shortcuts import render, redirect
from django.contrib.auth import login, logout, authenticate
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
from .models import Project, GeneratedDocument
from .utils.code_parser import parse_python_project
from .utils.doc_generator import generate_markdown_doc
import os, time
import pdfkit
from django.conf import settings
from .utils.doc_generator import generate_markdown_doc, generate_html_doc, generate_pdf_doc
from django.http import FileResponse, Http404, HttpResponse
import tempfile
from .forms import CustomUserCreationForm, CustomAuthenticationForm, ProjectUploadForm
from django.shortcuts import get_object_or_404
from django.core.files.base import ContentFile, File
from django.conf import settings
from django.core.exceptions import ObjectDoesNotExist
from django.contrib.auth.decorators import login_required
from django.db import transaction
import threading
from django.http import FileResponse
from django.shortcuts import get_object_or_404
from django.utils import timezone
from django.http import JsonResponse
from django.views.decorators.http import require_POST
from django.utils.timezone import make_aware
from datetime import datetime
from django.core.files import File
from google import genai
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_protect
from django.conf import settings
import json

# Configure Gemini with your API key
client = genai.Client(api_key=settings.GEMINI_API_KEY)

# Generate content
try:
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents="Analyze and explain best practices for writing clean, maintainable code documentation and provide examples of effective documentation patterns in Python"
    )
    print("Response:", response.text)
except Exception as e:
    print("Error:", str(e))

# Update the chat endpoint to use the new client
@csrf_protect
def chat_endpoint(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            message = data.get('message')
            
            # Get response from Gemini using the new client
            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=message
            )
            
            return JsonResponse({
                'response': response.text
            })
        except Exception as e:
            return JsonResponse({
                'error': str(e)
            }, status=500)
    return JsonResponse({'error': 'Method not allowed'}, status=405)

@csrf_protect
def upload_document(request):
    if request.method == 'POST':
        try:
            document = request.FILES['document']
            # Process the document using Gemini
            # Store the document content for future reference
            
            return JsonResponse({
                'message': 'Document uploaded successfully'
            })
        except Exception as e:
            return JsonResponse({
                'error': str(e)
            }, status=500)

def document_view(request, project_id, format_type):
    project = Project.objects.get(id=project_id, user=request.user)
    if not project.processed:
        return HttpResponse("Document not ready", status=400)
    
    project_path = os.path.join(settings.MEDIA_ROOT, project.zip_file.name)
    parsed_data = parse_python_project(project_path)
    
    if format_type == 'markdown':
        content = generate_markdown_doc(parsed_data)
        response = HttpResponse(content, content_type='text/markdown')
        response['Content-Disposition'] = f'attachment; filename="{project.name}_docs.md"'
    
    elif format_type == 'html':
        content = generate_html_doc(parsed_data)
        response = HttpResponse(content, content_type='text/html')
        response['Content-Disposition'] = f'attachment; filename="{project.name}_docs.html"'
    
    elif format_type == 'pdf':
        # Create temporary PDF file
        temp_pdf = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
        temp_pdf.close()
        generate_pdf_doc(parsed_data, temp_pdf.name)
        
        # Serve the PDF
        with open(temp_pdf.name, 'rb') as pdf:
            response = HttpResponse(pdf.read(), content_type='application/pdf')
            response['Content-Disposition'] = f'attachment; filename="{project.name}_docs.pdf"'
        
        # Clean up
        os.unlink(temp_pdf.name)
    
    else:
        return HttpResponse("Invalid format", status=400)
    
    return response

# Auth Views
def signup_view(request):
    if request.method == 'POST':
        form = UserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            return redirect('docs:dashboard')
    else:
        form = UserCreationForm()
    return render(request, 'docs/signup.html', {'form': form})

@login_required
def logout_view(request):
    logout(request)
    return redirect('docs:dashboard')  
# Document Preview
def preview_document(request, project_id):
    project = get_object_or_404(Project, id=project_id, user=request.user)
    
    if not project.processed:
        return HttpResponse("Document not ready", status=400)
    
    # Read the HTML file content
    with open(project.html_file.path, 'r') as f:
        html_content = f.read()
    
    return render(request, 'docs/document.html', {
        'project': project,
        'html_content': html_content
    })

def get_file_info(file_path):
    return {
        'name': os.path.basename(file_path),
        'size': os.path.getsize(file_path),
        'modified': make_aware(datetime.fromtimestamp(os.path.getmtime(file_path))),
        'url': os.path.join(settings.MEDIA_URL, file_path.replace(settings.MEDIA_ROOT + '/', ''))
    }

#listing uploaded zip files and generated documents per user
def dashboard(request):
    if not request.user.is_authenticated:
        return redirect('docs:login')

    try:
        # Get all projects for this user
        projects = Project.objects.filter(user=request.user).order_by('-upload_date')

        # Uploaded zip files
        uploaded_files = []
        for project in projects:
            if project.zip_file:
                uploaded_files.append({
                    'name': project.name,
                    'url': project.zip_file.url,
                    'size': project.zip_file.size,
                    'modified': project.upload_date
                })

        # Generated documents
        documentation_files = []
        for project in projects:
            generated_docs = GeneratedDocument.objects.filter(project=project)
            for doc in generated_docs:
                documentation_files.append({
                    'project_name': project.name,
                    'name': os.path.basename(doc.file.name),
                    'type': doc.doc_type,
                    'url': doc.file.url,
                    'size': doc.file.size,
                    'modified': doc.generated_at
                })

        return render(request, 'docs/dashboard.html', {
            'uploaded_files': uploaded_files,
            'documentation_files': sorted(documentation_files, key=lambda x: x['modified'], reverse=True),
            'user': request.user
        })

    except Exception as e:
        return render(request, 'docs/dashboard.html', {
            'error': str(e),
            'uploaded_files': [],
            'documentation_files': []
        })

def download_file(request, path):
    file_path = os.path.join(settings.MEDIA_ROOT, path)
    if os.path.exists(file_path):
        response = FileResponse(open(file_path, 'rb'), as_attachment=True)
        return response
    else:
        raise Http404("File not found")

@require_POST
def delete_file(request):
    if not request.user.is_authenticated:
        return JsonResponse({'status': 'error', 'message': 'Unauthorized'}, status=401)
    
    file_url = request.POST.get('file_url')
    file_type = request.POST.get('file_type')
    
    if not file_url or not file_type:
        return JsonResponse({'status': 'error', 'message': 'Missing parameters'}, status=400)
    
    try:
        # Find the project that owns this file
        project = None
        if file_type == 'html':
            project = Project.objects.filter(user=request.user, html_file=file_url).first()
        elif file_type == 'markdown':
            project = Project.objects.filter(user=request.user, markdown_file=file_url).first()
        elif file_type == 'pdf':
            project = Project.objects.filter(user=request.user, pdf_file=file_url).first()
        
        if not project:
            return JsonResponse({'status': 'error', 'message': 'File not found'}, status=404)
        
        # Delete the file field
        if file_type == 'html':
            project.html_file.delete()
        elif file_type == 'markdown':
            project.markdown_file.delete()
        elif file_type == 'pdf':
            project.pdf_file.delete()
        
        return JsonResponse({'status': 'success'})
        
    except Exception as e:
        return JsonResponse({'status': 'error', 'message': str(e)}, status=500)

def signup_view(request):
    if request.method == 'POST':
        form = CustomUserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            return redirect('docs:dashboard')
    else:
        form = CustomUserCreationForm()
    return render(request, 'docs/signup.html', {'form': form})

def login_view(request):
    if request.method == 'POST':
        form = CustomAuthenticationForm(request, data=request.POST)
        if form.is_valid():
            username = form.cleaned_data.get('username')
            password = form.cleaned_data.get('password')
            user = authenticate(username=username, password=password)
            if user is not None:
                login(request, user)
                return redirect('docs:dashboard')
    else:
        form = CustomAuthenticationForm()
    return render(request, 'docs/login.html', {'form': form})



def upload_project(request):
    if request.method == 'POST':
        form = ProjectUploadForm(request.POST, request.FILES)
        if form.is_valid():
            project = form.save(commit=False)
            project.user = request.user
            project.processed = False
            project.processing_error = None
            project.save()
            
            # Start background processing with proper project instance
            threading.Thread(
                target=process_project,
                args=(project.id,)  # Pass only the ID
            ).start()
            
            return redirect('docs:dashboard')
    else:
        form = ProjectUploadForm()
    return render(request, 'docs/upload.html', {'form': form})

#Process uploaded project files
def process_project(project_id):
    try:
        project = Project.objects.get(id=project_id)

        # Parse and generate documents
        parsed_data = parse_python_project(project.zip_file.path)

        # Directory for generated documents
        docs_dir = os.path.join(settings.MEDIA_ROOT, 'generated_docs')
        os.makedirs(docs_dir, exist_ok=True)

        # Markdown Generation
        md_content = generate_markdown_doc(parsed_data)
        md_filename = f"{project.name}_docs.md"
        md_path = os.path.join(docs_dir, md_filename)
        with open(md_path, 'w') as f:
            f.write(md_content)

        # HTML Generation
        html_content = generate_html_doc(parsed_data)
        html_filename = f"{project.name}_docs.html"
        html_path = os.path.join(docs_dir, html_filename)
        with open(html_path, 'w') as f:
            f.write(html_content)

        # PDF Generation (ensure wkhtmltopdf path is correct)
        pdf_filename = f"{project.name}_docs.pdf"
        pdf_path = os.path.join(docs_dir, pdf_filename)
        config = pdfkit.configuration(wkhtmltopdf=settings.PDFKIT_CONFIG['wkhtmltopdf'])
        options = {'encoding': 'UTF-8', 'quiet': ''}
        pdfkit.from_string(html_content, pdf_path, configuration=config, options=options)

        # Save the documents in the database inside a transaction block
        with transaction.atomic():
            # Save Markdown file
            with open(md_path, 'rb') as md_file:
                GeneratedDocument.objects.create(
                    project=project,
                    doc_type='markdown',
                    file=File(md_file, name=os.path.join('generated_docs', md_filename))
                )

            # Save HTML file
            with open(html_path, 'rb') as html_file:
                GeneratedDocument.objects.create(
                    project=project,
                    doc_type='html',
                    file=File(html_file, name=os.path.join('generated_docs', html_filename))
                )

            # Save PDF file
            with open(pdf_path, 'rb') as pdf_file:
                GeneratedDocument.objects.create(
                    project=project,
                    doc_type='pdf',
                    file=File(pdf_file, name=os.path.join('generated_docs', pdf_filename))
                )

            # Mark the project as processed
            project.processed = True
            project.save()

    except Exception as e:
        project.processed = False
        project.save()
        print(f"Error processing project {project.name}: {str(e)}")


def get_documentation_files(user):
    """Helper function to scan and return all documentation files for a user"""
    files = []
    doc_types = ['html', 'markdown']
    
    # Create user-specific 
    user_dir = str(user.id)  # or user.username, depending on your preference
    
    for doc_type in doc_types:
        dir_path = os.path.join(settings.MEDIA_ROOT, 'docs', doc_type, user_dir)
        if os.path.exists(dir_path):
            for filename in os.listdir(dir_path):
                file_path = os.path.join(dir_path, filename)
                if os.path.isfile(file_path):
                    stat = os.stat(file_path)
                    files.append({
                        'name': filename,
                        'type': doc_type,
                        'size': stat.st_size,
                        'modified': timezone.datetime.fromtimestamp(stat.st_mtime),
                        'url': os.path.join(settings.MEDIA_URL, 'docs', doc_type, user_dir, filename)
                    })
    
    return sorted(files, key=lambda x: x['modified'], reverse=True)

def download_doc(request, project_id, doc_type):
    project = get_object_or_404(Project, id=project_id, user=request.user)
    
    if doc_type == 'markdown' and project.markdown_file:
        response = FileResponse(project.markdown_file)
        response['Content-Disposition'] = f'attachment; filename="{project.name}_docs.md"'
    elif doc_type == 'html' and project.html_file:
        response = FileResponse(project.html_file)
        response['Content-Disposition'] = f'attachment; filename="{project.name}_docs.html"'
    elif doc_type == 'pdf' and project.pdf_file:
        response = FileResponse(project.pdf_file)
        response['Content-Disposition'] = f'attachment; filename="{project.name}_docs.pdf"'
    else:
        raise Http404("Document not available")
    
    return 

def download_document(request, project_id, doc_type):
    project = get_object_or_404(Project, id=project_id, user=request.user)
    
    if not project.processed:
        return HttpResponse("Document not ready", status=400)
    
    if doc_type == 'markdown' and project.markdown_path:
        file_path = os.path.join(settings.MEDIA_ROOT, project.markdown_path)
        if os.path.exists(file_path):
            return FileResponse(open(file_path, 'rb'), 
                             as_attachment=True,
                             filename=f"{project.name}_docs.md")
    
    elif doc_type == 'html' and project.html_path:
        file_path = os.path.join(settings.MEDIA_ROOT, project.html_path)
        if os.path.exists(file_path):
            return FileResponse(open(file_path, 'rb'),
                             as_attachment=True,
                             filename=f"{project.name}_docs.html")
    
    elif doc_type == 'pdf' and project.pdf_path:
        file_path = os.path.join(settings.MEDIA_ROOT, project.pdf_path)
        if os.path.exists(file_path):
            return FileResponse(open(file_path, 'rb'),
                             as_attachment=True,
                             filename=f"{project.name}_docs.pdf")
    
    return HttpResponse("Document not found", status=404)

#Download generated documents
def download_markdown(request, project_id):
    project = get_object_or_404(Project, id=project_id, user=request.user)
    return FileResponse(project.markdown_file, as_attachment=True, filename=f"{project.name}_docs.md")

def download_html(request, project_id):
    project = get_object_or_404(Project, id=project_id, user=request.user)
    return FileResponse(project.html_file, as_attachment=True, filename=f"{project.name}_docs.html")

def download_pdf(request, project_id):
    project = get_object_or_404(Project, id=project_id, user=request.user)
    return FileResponse(project.pdf_file, as_attachment=True, filename=f"{project.name}_docs.pdf")


def chat_view(request):
    return render(request, 'docs/chat.html')