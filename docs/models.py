from django.db import models
from django.contrib.auth.models import User
import os
from django.conf import settings

class Project(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    name = models.CharField(max_length=255)
    upload_date = models.DateTimeField(auto_now_add=True)
    zip_file = models.FileField(upload_to="projects/")
    processed = models.BooleanField(default=False)

    def __str__(self):
        return self.name

class GeneratedDocument(models.Model):
    DOC_TYPES = (
        ('markdown', 'Markdown'),
        ('html', 'HTML'),
        ('pdf', 'PDF'),
    )

    project = models.ForeignKey('Project', on_delete=models.CASCADE, related_name='documents')
    doc_type = models.CharField(max_length=10, choices=DOC_TYPES)
    file = models.FileField(upload_to='generated_docs/')
    generated_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-generated_at']

    def __str__(self):
        return f"{self.project.name} - {self.get_doc_type_display()}"
