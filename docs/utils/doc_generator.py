import pdfkit
from xhtml2pdf import pisa
import markdown2
from pathlib import Path
import os
from django.conf import settings

def generate_markdown_doc(parsed_data):
    """Generate comprehensive Markdown documentation with error handling"""
    if not parsed_data or not isinstance(parsed_data, list):
        return "# Documentation Generation Error\n\nNo valid parsed data provided"
    
    markdown = "# Project Documentation\n\n"
    
    for file_data in parsed_data:
        # Handle parse errors
        if file_data.get('error'):
            markdown += f"## ⚠️ Error in `{file_data['file']}`\n"
            markdown += f"{file_data['error']}\n\n"
            continue
            
        markdown += f"## File: `{file_data['file']}`\n\n"
        
        # Classes section
        if file_data.get('classes'):
            markdown += "### Classes\n\n"
            for cls in file_data['classes']:
                markdown += f"#### `{cls['name']}` (Line {cls.get('lineno', '?')})\n"
                markdown += f"{cls['docstring']}\n\n"
                
                if cls.get('methods'):
                    markdown += "**Methods:**\n"
                    for method in cls['methods']:
                        markdown += f"- `{method['name']}({', '.join(method['args'])})`  \n"
                        markdown += f"  *Line {method.get('lineno', '?')}: {method['docstring']}*\n\n"
        
        # Functions section
        if file_data.get('functions'):
            markdown += "### Functions\n\n"
            for func in file_data['functions']:
                markdown += f"#### `{func['name']}({', '.join(func['args'])})` (Line {func.get('lineno', '?')})\n"
                markdown += f"{func['docstring']}\n\n"
    
    return markdown

def generate_html_doc(parsed_data):
    """Generate styled HTML documentation with syntax highlighting"""
    markdown = generate_markdown_doc(parsed_data)
    html = markdown2.markdown(markdown, extras=["fenced-code-blocks"])
    
    return f"""<!DOCTYPE html>
<html>
<head>
    <title>Project Documentation</title>
    <meta charset="UTF-8">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.7.0/styles/atom-one-light.min.css" rel="stylesheet">
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 900px;
            margin: 0 auto;
            padding: 20px;
        }}
        h1 {{
            color: #6a0dad;
            border-bottom: 2px solid #f0f0f0;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #5a0a9d;
            margin-top: 30px;
            border-bottom: 1px dashed #e0e0e0;
        }}
        h3 {{
            color: #4a089d;
        }}
        code {{
            background: #f8f9fa;
            padding: 2px 5px;
            border-radius: 3px;
            font-family: 'Courier New', monospace;
        }}
        pre code {{
            display: block;
            padding: 15px;
            overflow-x: auto;
        }}
        .method-list {{
            padding-left: 20px;
        }}
        .error {{
            color: #dc3545;
            background: #fff0f0;
            padding: 10px;
            border-radius: 5px;
        }}
    </style>
</head>
<body>
    {html}
    <script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.7.0/highlight.min.js"></script>
    <script>hljs.highlightAll();</script>
</body>
</html>"""

def generate_pdf_doc(parsed_data, output_path):
    """Generate PDF with proper error handling and configuration"""
    html = generate_html_doc(parsed_data)
    
    # PDFKit configuration
    config = pdfkit.configuration(
        wkhtmltopdf=settings.PDFKIT_CONFIG.get('wkhtmltopdf', '')
    )
    
    options = {
        'encoding': 'UTF-8',
        'quiet': '',
        'margin-top': '15mm',
        'margin-right': '15mm',
        'margin-bottom': '15mm',
        'margin-left': '15mm',
        'print-media-type': '',
    }
    
    try:
        pdfkit.from_string(
            html,
            output_path,
            configuration=config,
            options=options
        )
        return True
    except Exception as e:
        print(f"PDFKit error: {str(e)}")
        try:
            with open(output_path, "wb") as f:
                pisa_status = pisa.CreatePDF(
                    html,
                    dest=f,
                    encoding='utf-8'
                )
            return not pisa_status.err
        except Exception as fallback_error:
            print(f"PDF fallback error: {str(fallback_error)}")
            return False