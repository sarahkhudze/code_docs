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

#### `DocumentationConfig` (Line 4)
No docstring

## File: `forms.py`

### Classes

#### `SearchForm` (Line 4)
No docstring

## File: `generators.py`

### Classes

#### `DocumentationGenerator` (Line 9)
No docstring

**Methods:**
- `__init__(self)`  
  *Line 10: No docstring*

- `generate(self, parsed_data, format)`  
  *Line 14: No docstring*

- `_generate_markdown(self, parsed_data)`  
  *Line 24: No docstring*

- `_generate_html(self, parsed_data)`  
  *Line 47: No docstring*

- `_generate_pdf(self, parsed_data)`  
  *Line 54: No docstring*

## File: `models.py`

### Classes

#### `Project` (Line 3)
No docstring

**Methods:**
- `__str__(self)`  
  *Line 9: No docstring*

#### `Documentation` (Line 12)
No docstring

**Methods:**
- `__str__(self)`  
  *Line 28: No docstring*

#### `Meta` (Line 25)
No docstring

## File: `parsers.py`

### Classes

#### `CodeParser` (Line 5)
No docstring

**Methods:**
- `__init__(self)`  
  *Line 6: No docstring*

- `parse_file(self, file_path)`  
  *Line 10: No docstring*

- `_parse_arguments(self, arguments)`  
  *Line 48: No docstring*

- `parse_directory(self, root_dir)`  
  *Line 76: No docstring*

## File: `serializers.py`

### Classes

#### `ProjectSerializer` (Line 4)
No docstring

#### `DocumentationSerializer` (Line 9)
No docstring

#### `Meta` (Line 5)
No docstring

#### `Meta` (Line 10)
No docstring

## File: `tests.py`

## File: `urls.py`

## File: `views.py`

### Classes

#### `HomeView` (Line 16)
No docstring

**Methods:**
- `get_context_data(self)`  
  *Line 18: No docstring*

#### `ProjectDetailView` (Line 36)
No docstring

**Methods:**
- `get_context_data(self)`  
  *Line 40: No docstring*

#### `ProjectViewSet` (Line 46)
No docstring

#### `GenerateDocsView` (Line 50)
No docstring

**Methods:**
- `get(self, request)`  
  *Line 53: No docstring*

#### `ExportDocumentationView` (Line 80)
No docstring

**Methods:**
- `get(self, request)`  
  *Line 81: No docstring*

- `_serve_pdf(self, documentation)`  
  *Line 112: No docstring*

- `_serve_text(self, documentation)`  
  *Line 116: No docstring*

#### `SearchView` (Line 120)
No docstring

**Methods:**
- `form_valid(self, form)`  
  *Line 124: No docstring*

#### `ProjectListView` (Line 130)
No docstring

#### `ProjectCreateView` (Line 134)
No docstring

#### `RecentDocsView` (Line 140)
No docstring

#### `PDFExportsView` (Line 145)
No docstring

#### `MarkdownView` (Line 150)
No docstring

**Methods:**
- `get_context_data(self)`  
  *Line 153: No docstring*

#### `AIAssistantView` (Line 159)
No docstring

**Methods:**
- `get_context_data(self)`  
  *Line 162: No docstring*

#### `SettingsView` (Line 167)
No docstring

**Methods:**
- `get_context_data(self)`  
  *Line 170: No docstring*

## File: `__init__.py`

## File: `0001_initial.py`

### Classes

#### `Migration` (Line 7)
No docstring

## File: `__init__.py`

