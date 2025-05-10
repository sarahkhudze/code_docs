from django import forms
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
from django.contrib.auth.models import User
from .models import Project

class CustomUserCreationForm(UserCreationForm):
    """
    Enhanced user registration form with email field and Bootstrap styling support
    """
    email = forms.EmailField(
        required=True,
        widget=forms.EmailInput(attrs={'class': 'form-control', 'placeholder': 'Email'})
    )
    username = forms.CharField(
        widget=forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'Username'})
    )
    password1 = forms.CharField(
        widget=forms.PasswordInput(attrs={'class': 'form-control', 'placeholder': 'Password'})
    )
    password2 = forms.CharField(
        widget=forms.PasswordInput(attrs={'class': 'form-control', 'placeholder': 'Confirm Password'})
    )

    class Meta:
        model = User
        fields = ('username', 'email', 'password1', 'password2')

    def clean_email(self):
        email = self.cleaned_data.get('email')
        if User.objects.filter(email=email).exists():
            raise forms.ValidationError("This email is already registered.")
        return email

class CustomAuthenticationForm(AuthenticationForm):
    """
    Custom login form with Bootstrap styling
    """
    username = forms.CharField(
        widget=forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'Username'})
    )
    password = forms.CharField(
        widget=forms.PasswordInput(attrs={'class': 'form-control', 'placeholder': 'Password'})
    )

class ProjectUploadForm(forms.ModelForm):
    """
    Form for uploading Python project ZIP files with validation
    """
    class Meta:
        model = Project
        fields = ('name', 'zip_file')
        widgets = {
            'name': forms.TextInput(attrs={
                'class': 'form-control',
                'placeholder': 'My Awesome Project'
            }),
            'zip_file': forms.FileInput(attrs={
                'class': 'form-control',
                'accept': '.zip'
            })
        }

    def clean_zip_file(self):
        zip_file = self.cleaned_data.get('zip_file')
        if not zip_file.name.endswith('.zip'):
            raise forms.ValidationError("Only ZIP archives are allowed.")
        return zip_file