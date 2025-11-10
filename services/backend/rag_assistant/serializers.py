from rest_framework import serializers

class RagQuerySerializer(serializers.Serializer):
    query = serializers.CharField(min_length=1, max_length=2000)
    system_id = serializers.IntegerField(min_value=1)
    context = serializers.JSONField(required=False, allow_null=True)
    max_results = serializers.IntegerField(min_value=1, max_value=10, default=3, required=False)

    def validate_query(self, value):
        if not value.strip():
            raise serializers.ValidationError("Текст запроса обязателен.")
        return value.strip()
