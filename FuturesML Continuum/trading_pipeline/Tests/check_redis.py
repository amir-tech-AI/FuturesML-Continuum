import json
import redis  # اگر قبلا نصب نکردی: pip install redis

# اتصال به Redis
r = redis.Redis(host='localhost', port=6379, db=0)

# خواندن داده‌ها
data_raw = r.get('features:BTCUSDT')  # به جای 'BTCUSDT' کلید خودت را بنویس

if data_raw:
    data_list = json.loads(data_raw)
    for i, item in enumerate(data_list[:5]):  # فقط ۵ رکورد اول
        print(i, item)
else:
    print("هیچ داده‌ای برای این کلید وجود ندارد.")
