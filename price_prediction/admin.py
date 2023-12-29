from django.contrib import admin

from .models import CustomUser
from .models import Commodity
from .models import HistoricalPrice
from .models import OverallTable
from .models import CommodityPrice
from .models import Potato_white



admin.site.register(CustomUser)
admin.site.register(Commodity)
admin.site.register(HistoricalPrice)
admin.site.register(OverallTable)
admin.site.register(CommodityPrice)
admin.site.register(Potato_white)

