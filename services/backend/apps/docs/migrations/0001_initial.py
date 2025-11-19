# Generated migration for docs app

from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    initial = True

    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
    ]

    operations = [
        migrations.CreateModel(
            name='DocumentCategory',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(help_text="Category name (e.g., 'Quick Start', 'API Reference')", max_length=100)),
                ('slug', models.SlugField(help_text='URL-friendly identifier (auto-generated from name)', max_length=100, unique=True)),
                ('icon', models.CharField(blank=True, help_text="Icon for category (emoji or SVG class, e.g., '🚀' or 'icon-rocket')", max_length=50)),
                ('description', models.TextField(blank=True, help_text='Brief description of this category')),
                ('order', models.IntegerField(default=0, help_text='Display order (lower numbers appear first)')),
                ('is_active', models.BooleanField(default=True, help_text='Whether this category is visible')),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('updated_at', models.DateTimeField(auto_now=True)),
            ],
            options={
                'verbose_name': 'Document Category',
                'verbose_name_plural': 'Document Categories',
                'ordering': ['order', 'name'],
            },
        ),
        migrations.CreateModel(
            name='Document',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('title', models.CharField(help_text='Document title (displayed as header)', max_length=200)),
                ('slug', models.SlugField(help_text='URL-friendly identifier (auto-generated from title)', max_length=200, unique=True)),
                ('summary', models.CharField(blank=True, help_text='Short summary for search results and previews', max_length=300)),
                ('content', models.TextField(help_text='Full document content in Markdown format')),
                ('tags', models.CharField(blank=True, help_text="Comma-separated tags for search (e.g., 'api, rest, authentication')", max_length=200)),
                ('order', models.IntegerField(default=0, help_text='Display order within category (lower numbers appear first)')),
                ('is_published', models.BooleanField(default=True, help_text='Whether this document is visible to users')),
                ('is_featured', models.BooleanField(default=False, help_text='Show this document prominently on the docs homepage')),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('updated_at', models.DateTimeField(auto_now=True)),
                ('view_count', models.PositiveIntegerField(default=0, help_text='Number of times this document has been viewed')),
                ('author', models.ForeignKey(blank=True, help_text='User who created this document', null=True, on_delete=django.db.models.deletion.SET_NULL, related_name='authored_documents', to=settings.AUTH_USER_MODEL)),
                ('category', models.ForeignKey(help_text='Category this document belongs to', on_delete=django.db.models.deletion.CASCADE, related_name='documents', to='docs.documentcategory')),
            ],
            options={
                'verbose_name': 'Document',
                'verbose_name_plural': 'Documents',
                'ordering': ['category__order', 'order', 'title'],
            },
        ),
        migrations.CreateModel(
            name='UserProgress',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('completed', models.BooleanField(default=False, help_text='Whether user marked this as completed')),
                ('last_viewed_at', models.DateTimeField(auto_now=True, help_text='Last time user viewed this document')),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('document', models.ForeignKey(help_text='Document that was viewed', on_delete=django.db.models.deletion.CASCADE, related_name='user_progress', to='docs.document')),
                ('user', models.ForeignKey(help_text='User tracking progress', on_delete=django.db.models.deletion.CASCADE, related_name='doc_progress', to=settings.AUTH_USER_MODEL)),
            ],
            options={
                'verbose_name': 'User Progress',
                'verbose_name_plural': 'User Progress Records',
                'unique_together': {('user', 'document')},
            },
        ),
        migrations.AddIndex(
            model_name='documentcategory',
            index=models.Index(fields=['slug'], name='docs_docume_slug_4e4a15_idx'),
        ),
        migrations.AddIndex(
            model_name='documentcategory',
            index=models.Index(fields=['order'], name='docs_docume_order_5b2c6a_idx'),
        ),
        migrations.AddIndex(
            model_name='document',
            index=models.Index(fields=['slug'], name='docs_docume_slug_7f8c3d_idx'),
        ),
        migrations.AddIndex(
            model_name='document',
            index=models.Index(fields=['category', 'order'], name='docs_docume_categor_8e5a2c_idx'),
        ),
        migrations.AddIndex(
            model_name='document',
            index=models.Index(fields=['is_published'], name='docs_docume_is_publ_9d4f1e_idx'),
        ),
        migrations.AddIndex(
            model_name='document',
            index=models.Index(fields=['is_featured'], name='docs_docume_is_feat_2a6b3c_idx'),
        ),
        migrations.AddIndex(
            model_name='userprogress',
            index=models.Index(fields=['user', 'completed'], name='docs_userpr_user_id_3c7e4d_idx'),
        ),
        migrations.AddIndex(
            model_name='userprogress',
            index=models.Index(fields=['last_viewed_at'], name='docs_userpr_last_vi_5d8f6e_idx'),
        ),
    ]
