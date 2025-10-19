import os
import sys
import django
from apps.rag_assistant.rag_core import default_local_orchestrator

# Добавляем текущую директорию в путь
sys.path.insert(0, '.')

# Минимальные переменные окружения
os.environ.setdefault('SECRET_KEY', 'test-key')
os.environ.setdefault('DEBUG', 'True')
os.environ.setdefault('DATABASE_NAME', 'test_db')
os.environ.setdefault('DATABASE_USER', 'test_user')
os.environ.setdefault('DATABASE_PASSWORD', 'test_pass')
os.environ.setdefault('DATABASE_HOST', 'localhost')
os.environ.setdefault('DATABASE_PORT', '5432')
os.environ.setdefault('REDIS_URL', 'redis://localhost:6379/1')
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.settings')

django.setup()


def main():
    print("🚀 Testing RAG core...")

    orch = default_local_orchestrator()

    docs = [
        "Hydraulic pressure is low, check pump operation and fluid levels",
        "Temperature sensor shows high readings, cooling system may need attention",
        "Oil contamination detected in hydraulic system, replace filters",
        "Pump failure suspected due to unusual noise and vibration patterns",
        "Pressure relief valve stuck open, system cannot maintain pressure"
    ]

    print("📦 Building embeddings and index...")
    path = orch.build_and_save(docs, version="test_v1", metadata={"test": True, "docs_count": len(docs)})
    print(f"✅ Index saved to: {path}")

    print("🔍 Testing search...")
    vindex = orch.load_index("test_v1")
    query_vectors = orch.embedder.encode(["pump problems", "pressure issues"])
    distances, indices = vindex.search(query_vectors, k=2)

    print("\n📋 Results:")
    print("Query 1 - 'pump problems':")
    for i, (idx, score) in enumerate(zip(indices[0], distances[0])):
        print(f"  {i+1}. Score: {score:.3f} - {docs[idx]}")

    print("\nQuery 2 - 'pressure issues':")
    for i, (idx, score) in enumerate(zip(indices[1], distances[1])):
        print(f"  {i+1}. Score: {score:.3f} - {docs[idx]}")

    print("\n🎉 RAG test completed successfully!")


if __name__ == "__main__":
    main()
