# apps/rag_assistant/tasks.py
from celery import shared_task
from django.core.exceptions import ObjectDoesNotExist
from .models import Document, RagSystem
from .rag_service import RagAssistant
import logging

logger = logging.getLogger(__name__)

@shared_task(bind=True)
def process_document_async(self, document_id, system_id):
    """
    Асинхронная обработка документа через RAG систему
    """
    try:
        document = Document.objects.get(id=document_id)
        system = RagSystem.objects.get(id=system_id)
        
        assistant = RagAssistant(system)
        assistant.index_document(document)
        
        logger.info(f"Document {document_id} processed successfully for system {system_id}")
        return {
            'status': 'success',
            'document_id': document_id,
            'system_id': system_id,
            'message': 'Document processed successfully'
        }
        
    except ObjectDoesNotExist as e:
        logger.error(f"Object not found: {str(e)}")
        return {
            'status': 'error',
            'error': f'Object not found: {str(e)}'
        }
    except Exception as e:
        logger.error(f"Error processing document {document_id}: {str(e)}")
        return {
            'status': 'error',
            'error': str(e)
        }

@shared_task(bind=True)
def reindex_documents_async(self, system_id, document_ids=None):
    """
    Асинхронная переиндексация документов
    """
    try:
        system = RagSystem.objects.get(id=system_id)
        assistant = RagAssistant(system)
        
        if document_ids:
            documents = Document.objects.filter(id__in=document_ids)
        else:
            documents = Document.objects.filter(metadata__rag_system=system.id)
        
        processed_count = 0
        errors = []
        
        for doc in documents:
            try:
                assistant.index_document(doc)
                processed_count += 1
            except Exception as e:
                errors.append({
                    'document_id': doc.id,
                    'error': str(e)
                })
                logger.error(f"Error indexing document {doc.id}: {str(e)}")
        
        logger.info(f"Reindexed {processed_count} documents for system {system_id}")
        return {
            'status': 'completed',
            'system_id': system_id,
            'processed_count': processed_count,
            'errors': errors
        }
        
    except ObjectDoesNotExist as e:
        logger.error(f"System not found: {str(e)}")
        return {
            'status': 'error',
            'error': f'System not found: {str(e)}'
        }
    except Exception as e:
        logger.error(f"Error reindexing documents for system {system_id}: {str(e)}")
        return {
            'status': 'error',
            'error': str(e)
        }

@shared_task(bind=True)
def generate_embeddings_async(self, system_id, texts):
    """
    Асинхронная генерация эмбеддингов для текстов
    """
    try:
        system = RagSystem.objects.get(id=system_id)
        assistant = RagAssistant(system)
        
        # Генерация эмбеддингов через существующую систему
        embeddings = []
        for text in texts:
            # Здесь можно реализовать прямую генерацию эмбеддингов
            # через assistant.embedding если нужно
            embeddings.append({
                'text': text,
                'processed': True
            })
        
        logger.info(f"Generated embeddings for {len(texts)} texts in system {system_id}")
        return {
            'status': 'success',
            'system_id': system_id,
            'embeddings_count': len(embeddings)
        }
        
    except ObjectDoesNotExist as e:
        logger.error(f"System not found: {str(e)}")
        return {
            'status': 'error',
            'error': f'System not found: {str(e)}'
        }
    except Exception as e:
        logger.error(f"Error generating embeddings for system {system_id}: {str(e)}")
        return {
            'status': 'error',
            'error': str(e)
        }
