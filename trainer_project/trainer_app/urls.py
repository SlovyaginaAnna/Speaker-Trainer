from django.urls import path
from . import views

urlpatterns = [
    path("register/", views.register),
    path("login/", views.login),
    path("logout/", views.logout),
    path("password_recovery/", views.password_recovery),
    path("password_update/", views.password_update),
    path("upload_file/", views.upload_file),
    path("archive/number_of_files/", views.archive_number_of_files),
    path("archive/file_info/", views.archive_file_info),
    path("archive/file_image/", views.archive_file_image),
    path("modified_file/", views.video_file),
    path("file_statistics/", views.file_statistics),
    path("statistics/", views.statistics),
    path("recommendations/", views.recommendations),
]
