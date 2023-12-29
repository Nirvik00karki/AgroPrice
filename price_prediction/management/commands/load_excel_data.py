import pandas as pd
from django.core.management.base import BaseCommand
from price_prediction.models import HistoricalPrice, Commodity

class Command(BaseCommand):
    help = 'Load data from preprocessed_data.xlsx into HistoricalPrice model'

    def handle(self, *args, **options):
        file_path = r'C:\Final year Project\AgroPrice\price_prediction\management\commands\cleaned_data.xlsx'

        try:
            df = pd.read_excel(file_path)

            for index, row in df.iterrows():
                # Retrieve or create a Commodity instance based on the commodity name
                commodity_instance, created = Commodity.objects.get_or_create(name=row['Commodity'])

                # Create HistoricalPrice instance with the Commodity instance
                HistoricalPrice.objects.create(
                    commodity=commodity_instance,
                    date=row['Date'],
                    unit=row['Unit'],
                    minimum=row['Minimum'],
                    maximum=row['Maximum'],
                    average=row['Average']
                    # Add other fields as needed
                )

            self.stdout.write('Data loaded successfully')

        except Exception as e:
            self.stderr.write(f'Error: {e}')
