"""Models for documentation system."""

from __future__ import annotations

from django.conf import settings
from django.db import models
from django.urls import reverse
from django.utils.text import slugify


class DocumentCategory(models.Model):
    """Category for organizing documentation."""

    name = models.CharField(
        max_length=100,
        help_text="Category name (e.g., 'Quick Start', 'API Reference')"
    )
    slug = models.SlugField(
        unique=True,
        max_length=100,
        help_text="URL-friendly identifier (auto-generated from name)"
    )
    icon = models.CharField(
        max_length=50,
        blank=True,
        help_text="Icon for category (emoji or SVG class, e.g., '🚀' or 'icon-rocket')"
    )
    description = models.TextField(
        blank=True,
        help_text="Brief description of this category"
    )
    order = models.IntegerField(
        default=0,
        help_text="Display order (lower numbers appear first)"
    )
    is_active = models.BooleanField(
        default=True,
        help_text="Whether this category is visible"
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["order", "name"]
        verbose_name = "Document Category"
        verbose_name_plural = "Document Categories"
        indexes = [
            models.Index(fields=["slug"]),
            models.Index(fields=["order"]),
        ]

    def __str__(self) -> str:
        icon_display = f"{self.icon} " if self.icon else ""
        return f"{icon_display}{self.name}"

    def save(self, *args, **kwargs) -> None:
        """Auto-generate slug from name if not provided."""
        if not self.slug:
            self.slug = slugify(self.name)
        super().save(*args, **kwargs)

    def get_absolute_url(self) -> str:
        """Return URL for this category."""
        return reverse("admin:docs_category", kwargs={"slug": self.slug})

    @property
    def document_count(self) -> int:
        """Get number of published documents in this category."""
        return self.documents.filter(is_published=True).count()


class Document(models.Model):
    """Documentation article/guide."""

    title = models.CharField(
        max_length=200,
        help_text="Document title (displayed as header)"
    )
    slug = models.SlugField(
        unique=True,
        max_length=200,
        help_text="URL-friendly identifier (auto-generated from title)"
    )
    category = models.ForeignKey(
        DocumentCategory,
        on_delete=models.CASCADE,
        related_name="documents",
        help_text="Category this document belongs to"
    )
    summary = models.CharField(
        max_length=300,
        blank=True,
        help_text="Short summary for search results and previews"
    )
    content = models.TextField(
        help_text="Full document content in Markdown format"
    )
    tags = models.CharField(
        max_length=200,
        blank=True,
        help_text="Comma-separated tags for search (e.g., 'api, rest, authentication')"
    )
    order = models.IntegerField(
        default=0,
        help_text="Display order within category (lower numbers appear first)"
    )
    is_published = models.BooleanField(
        default=True,
        help_text="Whether this document is visible to users"
    )
    is_featured = models.BooleanField(
        default=False,
        help_text="Show this document prominently on the docs homepage"
    )
    author = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="authored_documents",
        help_text="User who created this document"
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    view_count = models.PositiveIntegerField(
        default=0,
        help_text="Number of times this document has been viewed"
    )

    class Meta:
        ordering = ["category__order", "order", "title"]
        verbose_name = "Document"
        verbose_name_plural = "Documents"
        indexes = [
            models.Index(fields=["slug"]),
            models.Index(fields=["category", "order"]),
            models.Index(fields=["is_published"]),
            models.Index(fields=["is_featured"]),
        ]

    def __str__(self) -> str:
        return self.title

    def save(self, *args, **kwargs) -> None:
        """Auto-generate slug from title if not provided."""
        if not self.slug:
            self.slug = slugify(self.title)
        super().save(*args, **kwargs)

    def get_absolute_url(self) -> str:
        """Return URL for this document."""
        return reverse("admin:docs_detail", kwargs={"slug": self.slug})

    @property
    def tag_list(self) -> list[str]:
        """Return tags as a list."""
        return [tag.strip() for tag in self.tags.split(",") if tag.strip()]

    def increment_view_count(self) -> None:
        """Increment view counter."""
        self.view_count += 1
        self.save(update_fields=["view_count"])


class UserProgress(models.Model):
    """Track which documents a user has viewed/completed."""

    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="doc_progress",
        help_text="User tracking progress"
    )
    document = models.ForeignKey(
        Document,
        on_delete=models.CASCADE,
        related_name="user_progress",
        help_text="Document that was viewed"
    )
    completed = models.BooleanField(
        default=False,
        help_text="Whether user marked this as completed"
    )
    last_viewed_at = models.DateTimeField(
        auto_now=True,
        help_text="Last time user viewed this document"
    )
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        verbose_name = "User Progress"
        verbose_name_plural = "User Progress Records"
        unique_together = [("user", "document")]
        indexes = [
            models.Index(fields=["user", "completed"]),
            models.Index(fields=["last_viewed_at"]),
        ]

    def __str__(self) -> str:
        status = "✓" if self.completed else "○"
        return f"{status} {self.user.email} - {self.document.title}"
