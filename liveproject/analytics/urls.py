from django.urls import path
from . import views

app_name = "analytics"
urlpatterns = [
    path("", views.upload_csv,   name="upload"),
    path("results/", views.dashboard, name="dashboard"),
    path("download/", views.download_csv, name="download_csv"), 
    path("report/", views.generate_pdf, name="generate_report"),
]
