def build_event_probabilities(events):
    minute_buckets = {}  # e.g., { (1,15): [GOAL, SHOT], (16,30): [CARD], ... }

    for event in events:
        minute = event["minute"]
        key = (minute // 15) * 15 + 1
        bucket = (key, key + 14)

        if bucket not in minute_buckets:
            minute_buckets[bucket] = []

        minute_buckets[bucket].append(event["type"].lower())

    return minute_buckets