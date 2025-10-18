from apps.rag_assistant.models import RagSystem
from django.core.management.base import BaseCommand


class Command(BaseCommand):
    help = "Инициализация RAG-систем"

    def handle(self, *args, **options):
        RagSystem.objects.get_or_create(
            name="default",
            defaults={
                "description": "Default RAG",
                "model_name": "openai/gpt-3.5-turbo",
                "index_type": "faiss",
                "index_config": {},
            },
        )
        self.stdout.write(self.style.SUCCESS("RAG System initialized"))
