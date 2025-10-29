"""Initialize RAG system with sample hydraulic diagnostic documents."""

import json
from pathlib import Path
from django.core.management.base import BaseCommand
from apps.rag_assistant.models import RagSystem, Document
from apps.rag_assistant.rag_service import RAGService


class Command(BaseCommand):
    """Initialize RAG system with hydraulic diagnostic knowledge base."""
    
    help = "Initialize RAG system with sample hydraulic diagnostic documents"
    
    def add_arguments(self, parser):
        parser.add_argument(
            '--name',
            type=str,
            default='hydraulic-diagnostics-v1',
            help='Name for the RAG system'
        )
        parser.add_argument(
            '--rebuild',
            action='store_true',
            help='Rebuild existing RAG system'
        )
        parser.add_argument(
            '--load-samples',
            action='store_true',
            default=True,
            help='Load sample hydraulic documents'
        )
    
    def handle(self, *args, **options):
        rag_name = options['name']
        rebuild = options['rebuild']
        load_samples = options['load_samples']
        
        self.stdout.write(f"\n🤖 Initializing RAG system: {rag_name}\n")
        
        # Check if RAG system exists
        try:
            rag_system = RagSystem.objects.get(name=rag_name)
            if not rebuild:
                self.stdout.write(
                    self.style.WARNING(
                        f"RAG system '{rag_name}' already exists. Use --rebuild to recreate."
                    )
                )
                return
            else:
                self.stdout.write(
                    self.style.WARNING(f"Rebuilding existing RAG system '{rag_name}'...")
                )
                # Delete existing documents
                rag_system.documents.all().delete()
        except RagSystem.DoesNotExist:
            # Create new RAG system
            rag_system = RagSystem.objects.create(
                name=rag_name,
                description="Hydraulic diagnostic knowledge base with equipment troubleshooting guides",
                model_name="qwen2:7b",
                index_type="faiss",
                index_config={
                    "dimension": 768,
                    "metric": "inner_product",
                    "normalize_embeddings": True
                }
            )
            self.stdout.write(
                self.style.SUCCESS(f"Created new RAG system: {rag_name}")
            )
        
        # Load sample documents if requested
        if load_samples:
            self.load_sample_documents(rag_system)
        
        # Build RAG index
        self.build_rag_index(rag_system)
        
        self.stdout.write(
            self.style.SUCCESS(f"\n✅ RAG system '{rag_name}' initialized successfully!")
        )
        self.stdout.write(
            f"   Documents: {rag_system.documents.count()}"
        )
        self.stdout.write(
            f"   System ID: {rag_system.id}"
        )
        self.stdout.write(
            "\n📚 You can now query the RAG system via API: /api/rag/query/\n"
        )
    
    def load_sample_documents(self, rag_system):
        """Load sample hydraulic diagnostic documents."""
        self.stdout.write("Loading sample hydraulic diagnostic documents...")
        
        sample_documents = [
            {
                "title": "Hydraulic Pump Pressure Loss Diagnostics",
                "content": """Common causes of hydraulic pump pressure loss:
                
1. Internal leakage in pump components
   - Worn piston rings or cylinder bores
   - Damaged valve plates or port plates
   - Excessive clearances between moving parts
   
2. Contaminated hydraulic fluid
   - Dirt and particles causing wear
   - Water contamination reducing lubrication
   - Wrong viscosity affecting pump efficiency
   
3. Suction line problems
   - Air leaks in suction lines
   - Clogged suction strainers or filters
   - Insufficient fluid level in reservoir
   
Diagnostic steps:
- Measure system pressure at multiple points
- Check fluid condition and contamination levels
- Inspect suction lines for air leaks
- Monitor pump performance curves
- Analyze vibration and temperature patterns""",
                "format": "md",
                "language": "en",
                "metadata": {
                    "category": "pump_diagnostics",
                    "equipment_type": "hydraulic_pump",
                    "severity": "high",
                    "keywords": ["pressure_loss", "pump_failure", "diagnostics"]
                }
            },
            {
                "title": "Hydraulic System Overheating Solutions",
                "content": """Hydraulic system overheating causes and solutions:
                
Causes:
1. Excessive system pressure
   - Relief valve setting too high
   - Blocked return lines causing back pressure
   - Pump working against excessive load
   
2. Inadequate heat dissipation
   - Undersized or dirty heat exchanger
   - Low coolant flow or temperature
   - Insufficient reservoir size
   
3. High fluid viscosity
   - Wrong fluid specification
   - Cold weather operation
   - Contaminated or degraded fluid
   
Solutions:
- Install adequate cooling systems
- Check and adjust relief valve settings
- Clean heat exchangers regularly
- Monitor fluid temperature continuously
- Use proper viscosity hydraulic fluid
- Maintain adequate fluid levels""",
                "format": "md",
                "language": "en",
                "metadata": {
                    "category": "thermal_management",
                    "equipment_type": "hydraulic_system",
                    "severity": "medium",
                    "keywords": ["overheating", "temperature", "cooling"]
                }
            },
            {
                "title": "Диагностика гидравлических цилиндров",
                "content": """Основные неисправности гидроцилиндров:
                
1. Внутренние утечки:
   - Износ уплотнительных элементов
   - Повреждение цилиндра или поршня
   - Неправильная сборка узлов
   
2. Медленное движение штока:
   - Недостаточное давление рабочей жидкости
   - Загрязнение рабочей жидкости
   - Механическое заедание
   
Методы диагностики:
- Измерение давления в полостях цилиндра
- Контроль температуры рабочей жидкости
- Анализ скорости движения штока
- Проверка состояния уплотнителей""",
                "format": "md",
                "language": "ru",
                "metadata": {
                    "category": "cylinder_diagnostics",
                    "equipment_type": "hydraulic_cylinder",
                    "severity": "high",
                    "keywords": ["цилиндр", "утечка", "диагностика"]
                }
            }
        ]
        
        created_count = 0
        for doc_data in sample_documents:
            document = Document.objects.create(
                rag_system=rag_system,
                title=doc_data["title"],
                content=doc_data["content"],
                format=doc_data["format"],
                language=doc_data["language"],
                metadata=doc_data["metadata"]
            )
            created_count += 1
            self.stdout.write(f"  + {document.title}")
        
        self.stdout.write(
            self.style.SUCCESS(f"\nLoaded {created_count} sample documents")
        )
    
    def build_rag_index(self, rag_system):
        """Build FAISS index for the RAG system."""
        self.stdout.write("\nBuilding RAG index...")
        
        try:
            # For now, just acknowledge that documents are ready for indexing
            # The actual indexing will be handled by the RAG service
            documents_count = rag_system.documents.count()
            
            self.stdout.write(
                self.style.SUCCESS(f"RAG system ready with {documents_count} documents for indexing")
            )
            
        except Exception as e:
            self.stdout.write(
                self.style.ERROR(f"Error preparing RAG index: {str(e)}")
            )
            raise