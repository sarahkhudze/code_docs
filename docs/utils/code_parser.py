import ast
import os
import zipfile
import tempfile
from pathlib import Path

def parse_python_file(file_path):
    """Enhanced Python file parser with better error handling"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()
        
        tree = ast.parse(code)
        functions = []
        classes = []

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Skip if it's a method inside a class (we'll catch it in ClassDef)
                if any(isinstance(parent, ast.ClassDef) for parent in ast.walk(tree)):
                    continue
                    
                docstring = ast.get_docstring(node) or "No docstring"
                functions.append({
                    "name": node.name,
                    "args": [arg.arg for arg in node.args.args],
                    "docstring": docstring,
                    "lineno": node.lineno
                })
                
            elif isinstance(node, ast.ClassDef):
                methods = []
                for m in node.body:
                    if isinstance(m, ast.FunctionDef):
                        method_doc = ast.get_docstring(m) or "No docstring"
                        methods.append({
                            "name": m.name,
                            "args": [arg.arg for arg in m.args.args],
                            "docstring": method_doc,
                            "lineno": m.lineno
                        })
                
                classes.append({
                    "name": node.name,
                    "methods": methods,
                    "docstring": ast.get_docstring(node) or "No docstring",
                    "lineno": node.lineno
                })

        return {
            "file": str(Path(file_path).name),
            "functions": functions,
            "classes": classes,
            "error": None
        }
    except Exception as e:
        return {
            "file": str(Path(file_path).name),
            "error": f"Parse error: {str(e)}",
            "functions": [],
            "classes": []
        }

def parse_python_project(zip_path):
    """Handle ZIP extraction and parse all Python files"""
    parsed_data = []
    extracted_dir = tempfile.mkdtemp()
    
    try:
        # Extract ZIP
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extracted_dir)
        
        # Walk through extracted files
        for root, _, files in os.walk(extracted_dir):
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    parsed = parse_python_file(file_path)
                    if parsed:  # Only add if we got something
                        parsed_data.append(parsed)
                        
        return parsed_data
    finally:
        # Clean up extracted files
        for root, dirs, files in os.walk(extracted_dir, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))
        os.rmdir(extracted_dir)