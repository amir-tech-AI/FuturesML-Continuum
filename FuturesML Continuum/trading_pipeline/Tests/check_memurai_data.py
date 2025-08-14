# check_memurai_data_v2.py
import redis
import json

r = redis.Redis(host='localhost', port=6379, db=0)

keys = r.keys('features:*')
print("Keys in Memurai:", [k.decode() for k in keys])

for key in keys:
    key_str = key.decode()
    key_type = r.type(key).decode()
    print(f"\nKey: {key_str}")
    print(f"Type: {key_type}")

    if key_type == 'string':
        data = r.get(key)
        if data:
            sample = json.loads(data.decode())
            print("Sample data:", sample)
            print("Number of entries: 1 (stored as single JSON string)")
    elif key_type == 'list':
        data = r.lrange(key, 0, -1)
        print("Number of entries:", len(data))
        if len(data) > 0:
            sample = json.loads(data[0].decode())
            print("Sample entry:", sample)
    else:
        print("Unhandled key type.")
