from recommender import Model1, Model2

song_ratings = [
    {"track_id": "0e7ipj03S05BNilyu5bRzt", "track_name": "rockstar (feat. 21 Savage)", "rating": 4},
    {"track_id": "5qaEfEh1AtSdrdrByCP7qR", "track_name": "Demons", "rating": 3},
    {"track_id": "7KXjTSCq5nL1LoYtL7XAwS", "track_name": "HUMBLE.", "rating": 5},
    {"track_id": "7BqHUALzNBTanL6OvsqmC1", "track_name": "Happier", "rating": 4},
    {"track_id": "7m9OqQk4RVRkw9JJdeAw96", "track_name": "Jocelyn Flores", "rating": 3},
    {"track_id": "3bNv3VuUOKgrf5hu3YcuRo", "track_name": "Someone Like You", "rating": 5},
    {"track_id": "2k1yPYf9WGA4LiqcLVwtzn", "track_name": "Another One Bites The Dust", "rating": 5},
    {"track_id": "2grjqo0Frpf2okIBiifQKs", "track_name": "September", "rating": 2},
    {"track_id": "2iUXsYOEPhVqEBwsqP70rE", "track_name": "Youngblood", "rating": 2},
    {"track_id": "1CmUZGtH29Kx36C1Hleqlz", "track_name": "Thrift Shop (feat. Wanz)", "rating": 4},
    {"track_id": "5DxXgozhkPLgrbKFY91w0c", "track_name": "Vete", "rating": 1},
    {"track_id": "3l9CW99AHtExIRV4hW2N5m", "track_name": "Misery Business", "rating": 2},
    {"track_id": "1jF7IL57ayN4Ity3jQqGu0", "track_name": "Try", "rating": 2},
    {"track_id": "5QpaGzWp0hwB5faV8dkbAz", "track_name": "Wherever You Will Go", "rating": 4},
    {"track_id": "2q4rjDy9WhaN3o9MvDbO21", "track_name": "Kiss Me Thru The Phone", "rating": 1},
    {"track_id": "1oew3nFNY3vMacJAsvry0S", "track_name": "Me And My Broken Heart", "rating": 4},
    {"track_id": "0mEdbdeRFQwBhN4xfyIeUM", "track_name": "Can't Tell Me Nothing", "rating": 1},
    {"track_id": "30QR0ndUdiiMQMA9g1PGCm", "track_name": "...And to Those I Love, Thanks for Sticking Around", "rating": 2},
    {"track_id": "009ImBOrIUlWgla8U05RAC", "track_name": "Unconditionally", "rating": 2},
    {"track_id": "40FUdLENDY3sZmHEM25lpE", "track_name": "Paradise", "rating": 4},
]

model1 = Model1()
model2 = Model2()

print("=== MODEL 1 ===")
print(model1.query(song_ratings, topk=5))

print("=== MODEL 2 ===")
print(model2.query(song_ratings, topk=5))