# Project Documentation

## File: `manage.py`

### Functions

#### `main()` (Line 7)
Run administrative tasks.

## File: `asgi.py`

## File: `settings.py`

## File: `urls.py`

## File: `wsgi.py`

## File: `__init__.py`

## File: `admin.py`

## File: `apps.py`

### Classes

#### `DocsConfig` (Line 4)
No docstring

## File: `forms.py`

### Classes

#### `CustomUserCreationForm` (Line 6)
Enhanced user registration form with email field and Bootstrap styling support

**Methods:**
- `clean_email(self)`  
  *Line 28: No docstring*

#### `CustomAuthenticationForm` (Line 34)
Custom login form with Bootstrap styling

#### `ProjectUploadForm` (Line 45)
Form for uploading Python project ZIP files with validation

**Methods:**
- `clean_zip_file(self)`  
  *Line 63: No docstring*

#### `Meta` (Line 24)
No docstring

#### `Meta` (Line 49)
No docstring

## File: `models.py`

### Classes

#### `Project` (Line 6)
No docstring

**Methods:**
- `__str__(self)`  
  *Line 13: No docstring*

#### `GeneratedDocument` (Line 16)
No docstring

**Methods:**
- `__str__(self)`  
  *Line 31: No docstring*

#### `Meta` (Line 28)
No docstring

## File: `tests.py`

## File: `urls.py`

## File: `views.py`

### Functions

#### `document_view(request, project_id, format_type)` (Line 34)
No docstring

#### `signup_view(request)` (Line 84)
No docstring

#### `logout_view(request)` (Line 96)
No docstring

#### `preview_document(request, project_id)` (Line 100)
No docstring

#### `get_file_info(file_path)` (Line 115)
No docstring

#### `dashboard(request)` (Line 123)
No docstring

#### `dashboard(request)` (Line 168)
No docstring

#### `download_file(request, path)` (Line 214)
No docstring

#### `delete_file(request)` (Line 223)
No docstring

#### `signup_view(request)` (Line 259)
No docstring

#### `login_view(request)` (Line 270)
No docstring

#### `upload_project(request)` (Line 286)
No docstring

#### `process_project(project_id)` (Line 308)
No docstring

#### `get_documentation_files(user)` (Line 376)
Helper function to scan and return all documentation files for a user

#### `download_doc(request, project_id, doc_type)` (Line 401)
No docstring

#### `download_document(request, project_id, doc_type)` (Line 418)
No docstring

#### `download_markdown(request, project_id)` (Line 448)
No docstring

#### `download_html(request, project_id)` (Line 452)
No docstring

#### `download_pdf(request, project_id)` (Line 456)
No docstring

## File: `__init__.py`

## File: `0001_initial.py`

### Classes

#### `Migration` (Line 8)
No docstring

## File: `0002_project_html_file_project_markdown_file_and_more.py`

### Classes

#### `Migration` (Line 6)
No docstring

## File: `0003_generateddocument.py`

### Classes

#### `Migration` (Line 7)
No docstring

## File: `0004_remove_project_doc_file_remove_project_html_file_and_more.py`

### Classes

#### `Migration` (Line 6)
No docstring

## File: `__init__.py`

## File: `code_parser.py`

### Functions

#### `parse_python_file(file_path)` (Line 7)
Enhanced Python file parser with better error handling

#### `parse_python_project(zip_path)` (Line 64)
Handle ZIP extraction and parse all Python files

## File: `doc_generator.py`

### Functions

#### `generate_markdown_doc(parsed_data)` (Line 8)
Generate comprehensive Markdown documentation with error handling

#### `generate_html_doc(parsed_data)` (Line 46)
Generate styled HTML documentation with syntax highlighting

#### `generate_pdf_doc(parsed_data, output_path)` (Line 108)
Generate PDF with proper error handling and configuration

## File: `pdf_utils.py`

### Functions

#### `get_wkhtmltopdf_path()` (Line 4)
No docstring

