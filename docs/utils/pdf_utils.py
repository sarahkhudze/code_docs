import os
from django.conf import settings

def get_wkhtmltopdf_path():
    # Try to find wkhtmltopdf in common locations
    possible_paths = [
        '/usr/local/bin/wkhtmltopdf',
        '/usr/bin/wkhtmltopdf',
        os.path.join(os.getcwd(), 'wkhtmltopdf', 'bin', 'wkhtmltopdf.exe')
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return path
    
    return None

WKHTMLTOPDF_PATH = get_wkhtmltopdf_path()