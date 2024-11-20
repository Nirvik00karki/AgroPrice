from rest_framework import serializers
from .models import HistoricalPrice
from .models import OverallTable

class HistoricalPriceSerializer(serializers.ModelSerializer):
    class Meta:
        model = HistoricalPrice
        fields = '__all__'

class OverallTableSerializer(serializers.ModelSerializer):
    class Meta:
        model = OverallTable
        fields = '__all__' 
