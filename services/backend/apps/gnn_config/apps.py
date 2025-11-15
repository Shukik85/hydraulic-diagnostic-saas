"""AppConfig for GNN Configuration/Training management."""
from django.apps import AppConfig
class GNNConfigConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "apps.gnn_config"
    verbose_name = "GNN Configuration"
