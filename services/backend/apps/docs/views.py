"""Views for documentation system."""

from __future__ import annotations

from django.contrib.admin.views.decorators import staff_member_required
from django.db.models import Q
from django.http import JsonResponse
from django.shortcuts import get_object_or_404, render
from django.views.decorators.http import require_POST

from .models import Document, DocumentCategory, UserProgress


@staff_member_required
def docs_index(request):
    """Display documentation homepage with categories and featured docs."""
    categories = DocumentCategory.objects.filter(is_active=True).prefetch_related("documents")

    featured_docs = Document.objects.filter(is_published=True, is_featured=True).select_related(
        "category"
    )[:6]

    # Get user progress if authenticated
    user_progress = {}
    if request.user.is_authenticated:
        progress_qs = UserProgress.objects.filter(user=request.user, completed=True).values_list(
            "document_id", flat=True
        )
        user_progress = set(progress_qs)

    context = {
        "categories": categories,
        "featured_docs": featured_docs,
        "user_progress": user_progress,
    }
    return render(request, "docs/index.html", context)


@staff_member_required
def docs_category(request, slug: str):
    """Display all documents in a specific category."""
    category = get_object_or_404(DocumentCategory, slug=slug, is_active=True)

    documents = Document.objects.filter(category=category, is_published=True).select_related(
        "author"
    )

    # Get user progress
    user_progress = set()
    if request.user.is_authenticated:
        progress_qs = UserProgress.objects.filter(
            user=request.user, document__category=category, completed=True
        ).values_list("document_id", flat=True)
        user_progress = set(progress_qs)

    context = {
        "category": category,
        "documents": documents,
        "user_progress": user_progress,
    }
    return render(request, "docs/category.html", context)


@staff_member_required
def docs_detail(request, slug: str):
    """Display individual document with full content."""
    document = get_object_or_404(Document, slug=slug, is_published=True)

    # Increment view count
    document.increment_view_count()

    # Track user progress
    user_completed = False
    if request.user.is_authenticated:
        progress, _ = UserProgress.objects.get_or_create(user=request.user, document=document)
        user_completed = progress.completed

    # Get related documents from same category
    related_docs = Document.objects.filter(category=document.category, is_published=True).exclude(
        id=document.id
    )[:5]

    context = {
        "document": document,
        "user_completed": user_completed,
        "related_docs": related_docs,
    }
    return render(request, "docs/detail.html", context)


@staff_member_required
def docs_search(request):
    """Search documentation by title, content, or tags."""
    query = request.GET.get("q", "").strip()

    results = []
    if query:
        results = Document.objects.filter(
            Q(title__icontains=query)
            | Q(summary__icontains=query)
            | Q(content__icontains=query)
            | Q(tags__icontains=query),
            is_published=True,
        ).select_related("category")[:50]

    context = {
        "query": query,
        "results": results,
        "result_count": len(results),
    }
    return render(request, "docs/search.html", context)


@staff_member_required
@require_POST
def mark_complete(request, slug: str):
    """Mark a document as completed by the user (AJAX endpoint)."""
    document = get_object_or_404(Document, slug=slug)

    if not request.user.is_authenticated:
        return JsonResponse({"error": "Authentication required"}, status=401)

    progress, _ = UserProgress.objects.get_or_create(user=request.user, document=document)

    # Toggle completion status
    progress.completed = not progress.completed
    progress.save()

    return JsonResponse(
        {
            "success": True,
            "completed": progress.completed,
            "message": "Marked as completed" if progress.completed else "Marked as incomplete",
        }
    )
