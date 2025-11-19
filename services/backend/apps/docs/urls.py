"""URL configuration for docs app."""

from django.urls import path

from . import views

app_name = "docs"

urlpatterns = [
    path("", views.docs_index, name="index"),
    path("search/", views.docs_search, name="search"),
    path("category/<slug:slug>/", views.docs_category, name="category"),
    path("doc/<slug:slug>/", views.docs_detail, name="detail"),
    path("api/mark-complete/<slug:slug>/", views.mark_complete, name="mark_complete"),
]
