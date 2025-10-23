"""Модуль проекта с автогенерированным докстрингом."""

from django.core.management.base import BaseCommand

from apps.rag_assistant.models import Document, RagSystem
from apps.rag_assistant.rag_service import RagAssistant


class Command(BaseCommand):
    help = "Переиндексация документов для всех систем"

    def handle(self, *args, **options):
        for system in RagSystem.objects.all():
            assistant = RagAssistant(system)
            docs = Document.objects.all()
            for doc in docs:
                assistant.index_document(doc)
            self.stdout.write(f"Reindexed for system {system.name}")
