from django.core.management.base import BaseCommand
from apps.rag_assistant.models import KnowledgeBase, DocumentChunk
from apps.rag_assistant.rag_service import DocumentProcessor
import logging

logger = logging.getLogger(__name__)

class Command(BaseCommand):
    help = 'Переиндексация документов в базе знаний'
    
    def add_arguments(self, parser):
        parser.add_argument(
            '--document-id',
            type=str,
            help='ID конкретного документа для переиндексации',
        )
        parser.add_argument(
            '--all',
            action='store_true',
            help='Переиндексировать все документы',
        )
        parser.add_argument(
            '--category',
            type=str,
            choices=[choice for choice in KnowledgeBase.CATEGORY_CHOICES],
            help='Переиндексировать документы конкретной категории',
        )
    
    def handle(self, *args, **options):
        processor = DocumentProcessor()
        
        if options['document_id']:
            self._reindex_document(processor, options['document_id'])
        elif options['all']:
            self._reindex_all_documents(processor)
        elif options['category']:
            self._reindex_category(processor, options['category'])
        else:
            self.stdout.write(
                self.style.ERROR('Укажите --document-id, --all или --category')
            )
    
    def _reindex_document(self, processor, document_id):
        """Переиндексация конкретного документа"""
        try:
            document = KnowledgeBase.objects.get(id=document_id)
            self.stdout.write(f'🔄 Переиндексация: {document.title}')
            
            processor.reprocess_document(document)
            
            self.stdout.write(
                self.style.SUCCESS(f'✅ Документ переиндексирован: {document.title}')
            )
        except KnowledgeBase.DoesNotExist:
            self.stdout.write(
                self.style.ERROR(f'❌ Документ с ID {document_id} не найден')
            )
        except Exception as e:
            self.stdout.write(
                self.style.ERROR(f'❌ Ошибка переиндексации: {e}')
            )
    
    def _reindex_all_documents(self, processor):
        """Переиндексация всех документов"""
        documents = KnowledgeBase.objects.filter(status='active')
        total = documents.count()
        
        self.stdout.write(f'🔄 Переиндексация {total} документов...')
        
        success_count = 0
        error_count = 0
        
        for i, document in enumerate(documents, 1):
            try:
                self.stdout.write(f'[{i}/{total}] {document.title}')
                processor.reprocess_document(document)
                success_count += 1
                
            except Exception as e:
                error_count += 1
                self.stdout.write(
                    self.style.ERROR(f'❌ Ошибка: {document.title} - {e}')
                )
        
        self.stdout.write(
            self.style.SUCCESS(
                f'✅ Завершено. Успешно: {success_count}, Ошибки: {error_count}'
            )
        )
    
    def _reindex_category(self, processor, category):
        """Переиндексация документов категории"""
        documents = KnowledgeBase.objects.filter(
            status='active', 
            category=category
        )
        total = documents.count()
        
        category_display = dict(KnowledgeBase.CATEGORY_CHOICES)[category]
        self.stdout.write(f'🔄 Переиндексация категории "{category_display}" ({total} документов)...')
        
        success_count = 0
        error_count = 0
        
        for i, document in enumerate(documents, 1):
            try:
                self.stdout.write(f'[{i}/{total}] {document.title}')
                processor.reprocess_document(document)
                success_count += 1
                
            except Exception as e:
                error_count += 1
                self.stdout.write(
                    self.style.ERROR(f'❌ Ошибка: {document.title} - {e}')
                )
        
        self.stdout.write(
            self.style.SUCCESS(
                f'✅ Завершено. Успешно: {success_count}, Ошибки: {error_count}'
            )
        )
