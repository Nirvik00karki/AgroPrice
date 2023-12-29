from django.core.management.base import BaseCommand
from price_prediction.models import Commodity

class Command(BaseCommand):
    help = 'Populate the Commodity model with unique commodity names'

    def handle(self, *args, **kwargs):
        unique_commodities = [
            'Onion Dry (Indian)',
            'Tomato Big(Nepali)',
            'Potato White',
            'Garlic Dry Nepali',
            'Carrot(Local)',
            'Lettuce',
            'Apple(Jholey)',
            'Banana',
            'Cucumber(Local)'

        ]

        for commodity_name in unique_commodities:
            Commodity.objects.get_or_create(name=commodity_name)

        self.stdout.write('Commodities populated successfully')
