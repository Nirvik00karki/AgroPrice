from django.db import models
from django.contrib.auth.models import AbstractUser



class CustomUser(AbstractUser):
    username = models.CharField(max_length=150, unique=True)
    email = models.EmailField(unique=True)

    groups = models.ManyToManyField(
        'auth.Group',
        related_name='custom_user_groups',
        related_query_name='user',
        blank=True,
        help_text='The groups this user belongs to. A user will get all permissions granted to each of their groups.',
        verbose_name='groups',
    )

    user_permissions = models.ManyToManyField(
        'auth.Permission',
        related_name='custom_user_permissions',
        related_query_name='user',
        blank=True,
        help_text='Specific permissions for this user.',
        verbose_name='user permissions',
    )

    def __str__(self):
        return f'{self.username}'
    
class Commodity(models.Model):
    id = models.AutoField(primary_key=True)
    name = models.CharField(max_length=255, unique=True)

class HistoricalPrice(models.Model):
    SEASON_CHOICES = [
        ('Spring', 'Spring'),
        ('Summer', 'Summer'),
        ('Fall', 'Fall'),
        ('Winter', 'Winter'),
    ]
    commodity = models.ForeignKey(Commodity, on_delete=models.CASCADE)
    date = models.DateField()
    unit = models.CharField(max_length=50, default=None)
    minimum = models.DecimalField(max_digits=10, decimal_places=2)
    maximum = models.DecimalField(max_digits=10, decimal_places=2)
    average = models.DecimalField(max_digits=10, decimal_places=2)
    season = models.CharField(max_length=10, choices=SEASON_CHOICES)

    def __str__(self):
        return f"{self.commodity} - {self.date} ({self.season})"    
    
class OverallTable(models.Model):
    commodity = models.CharField(max_length=255)
    percentage_change = models.FloatField()

    def __str__(self):
        return self.commodity


class CommodityPrice(models.Model):
    commodity = models.CharField(max_length=100)
    date = models.DateField()
    average = models.FloatField()
    day = models.FloatField()
    month = models.FloatField()
    year = models.FloatField()
    season_fall = models.BooleanField()
    season_spring = models.BooleanField()
    season_summer = models.BooleanField()
    season_winter = models.BooleanField()
    apple_jholey = models.BooleanField()
    banana = models.BooleanField()
    carrot_local = models.BooleanField()
    cucumber_local = models.BooleanField()
    garlic_dry_nepali = models.BooleanField()
    lettuce = models.BooleanField()
    onion_dry_indian = models.BooleanField()
    potato_white = models.BooleanField()
    tomato_big_nepali = models.BooleanField()

    def __str__(self):
        return f"{self.commodity} - {self.date}"


class Potato_white(models.Model):
    Commodity = models.CharField(max_length=100)
    Date = models.DateTimeField()
    Average = models.FloatField()
    day = models.FloatField()
    month = models.FloatField()
    year = models.FloatField()
    Season_Fall = models.BooleanField()
    Season_Spring = models.BooleanField()
    Season_Summer = models.BooleanField()
    Season_Winter = models.BooleanField()
    Apple_Jholey = models.BooleanField()
    Banana = models.BooleanField()
    Carrot_Local = models.BooleanField()
    Cucumber_Local = models.BooleanField()
    Garlic_Dry_Nepali = models.BooleanField()
    Lettuce = models.BooleanField()
    Onion_Dry_Indian = models.BooleanField()
    Potato_White = models.BooleanField()
    Tomato_Big_Nepali = models.BooleanField()
    Festival_Buddha_Jayanti = models.BooleanField()
    Festival_Dashain = models.BooleanField()
    Festival_Gai_Jatra = models.BooleanField()
    Festival_Ghode_Jatra = models.BooleanField()
    Festival_Holi = models.BooleanField()
    Festival_Indra_Jatra = models.BooleanField()
    Festival_Janai_Purnima = models.BooleanField()
    Festival_Lhosar = models.BooleanField()
    Festival_Maghe_Sankranti = models.BooleanField()
    Festival_Maha_Shivaratri = models.BooleanField()
    Festival_Shree_Panchami = models.BooleanField()
    Festival_Teej = models.BooleanField()
    Festival_Tihar = models.BooleanField()
    Festival_nan = models.BooleanField()
    Dashain_near = models.IntegerField()
    Tihar_near = models.IntegerField()
    Holi_near = models.IntegerField()
    Maha_Shivaratri_near = models.IntegerField()
    Buddha_Jayanti_near = models.IntegerField()
    Ghode_Jatra_near = models.IntegerField()
    Teej_near = models.IntegerField()
    Indra_Jatra_near = models.IntegerField()
    Lhosar_near = models.IntegerField()
    Janai_Purnima_near = models.IntegerField()
    Gai_Jatra_near = models.IntegerField()
    Maghe_Sankranti_near = models.IntegerField()
    Shree_Panchami_near = models.IntegerField()
    Fall_near = models.IntegerField()
    Spring_near = models.IntegerField()
    Summer_near = models.IntegerField()
    Winter_near = models.IntegerField()

    def __str__(self):
        return f"{self.Commodity} - {self.Date}"

