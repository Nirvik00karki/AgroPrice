# Define a management command to import data from CSV
import csv
from django.core.management.base import BaseCommand
from price_prediction.models import Potato_white  

class Command(BaseCommand):
    help = 'Load data from CSV into Potato_white model'

    def handle(self, *args, **kwargs):
        file_path = r'C:\Users\nirvi\OneDrive\Desktop\Programs\Mlcurrent\potato_white.csv' 

        with open(file_path, 'r') as csv_file:
            csv_reader = csv.DictReader(csv_file)
            for row in csv_reader:
                # Map CSV columns to model fields and create Potato_white objects
                Potato_white.objects.create(
                    Commodity=row['Commodity'],
                    Date=row['Date'],
                    Average=float(row['Average']),
                    day=float(row['day']),
                    month=float(row['month']),
                    year=float(row['year']),
                    Season_Fall=row['Season_Fall'].lower() == 'true',
                    Season_Spring=row['Season_Spring'].lower() == 'true',
                    Season_Summer=row['Season_Summer'].lower() == 'true',
                    Season_Winter=row['Season_Winter'].lower() == 'true',
                    Apple_Jholey=row['Apple(Jholey)'].lower() == 'true',
                    Banana=row['Banana'].lower() == 'true',
                    Carrot_Local=row['Carrot(Local)'].lower() == 'true',
                    Cucumber_Local=row['Cucumber(Local)'].lower() == 'true',
                    Garlic_Dry_Nepali=row['Garlic Dry Nepali'].lower() == 'true',
                    Lettuce=row['Lettuce'].lower() == 'true',
                    Onion_Dry_Indian=row['Onion Dry (Indian)'].lower() == 'true',
                    Potato_White=row['Potato White'].lower() == 'true',
                    Tomato_Big_Nepali=row['Tomato Big(Nepali)'].lower() == 'true',
                    Festival_Buddha_Jayanti=row['Festival_Buddha Jayanti'].lower() == 'true',
                    Festival_Dashain=row['Festival_Dashain'].lower() == 'true',
                    Festival_Gai_Jatra=row['Festival_Gai Jatra'].lower() == 'true',
                    Festival_Ghode_Jatra=row['Festival_Ghode Jatra'].lower() == 'true',
                    Festival_Holi=row['Festival_Holi'].lower() == 'true',
                    Festival_Indra_Jatra=row['Festival_Indra Jatra'].lower() == 'true',
                    Festival_Janai_Purnima=row['Festival_Janai Purnima'].lower() == 'true',
                    Festival_Lhosar=row['Festival_Lhosar'].lower() == 'true',
                    Festival_Maghe_Sankranti=row['Festival_Maghe Sankranti'].lower() == 'true',
                    Festival_Maha_Shivaratri=row['Festival_Maha Shivaratri'].lower() == 'true',
                    Festival_Shree_Panchami=row['Festival_Shree Panchami'].lower() == 'true',
                    Festival_Teej=row['Festival_Teej'].lower() == 'true',
                    Festival_Tihar=row['Festival_Tihar'].lower() == 'true',
                    Festival_nan=row['Festival_nan'].lower() == 'true',
                    Dashain_near=int(row['Dashain_near']),
                    Tihar_near=int(row['Tihar_near']),
                    Holi_near=int(row['Holi_near']),
                    Maha_Shivaratri_near=int(row['Maha Shivaratri_near']),
                    Buddha_Jayanti_near=int(row['Buddha Jayanti_near']),
                    Ghode_Jatra_near=int(row['Ghode Jatra_near']),
                    Teej_near=int(row['Teej_near']),
                    Indra_Jatra_near=int(row['Indra Jatra_near']),
                    Lhosar_near=int(row['Lhosar_near']),
                    Janai_Purnima_near=int(row['Janai Purnima_near']),
                    Gai_Jatra_near=int(row['Gai Jatra_near']),
                    Maghe_Sankranti_near=int(row['Maghe Sankranti_near']),
                    Shree_Panchami_near=int(row['Shree Panchami_near']),
                    Fall_near=int(row['Fall_near']),
                    Spring_near=int(row['Spring_near']),
                    Summer_near=int(row['Summer_near']),
                    Winter_near=int(row['Winter_near']),
                )
        self.stdout.write('Data imported successfully')