# import os
# from xml.dom import NotFoundErr
# # import geopy.distance

# from pymongo import MongoClient


# class Database:
#     def init(self):
#         self.client = MongoClient("mongodb://localhost:27017")
#         self.db = self.client["umai"]
#         self.detections = self.db["detections"]

#     def insert(self, table, data):
#         if type(data) == list:
#             result = table.insert_many(data)
#             for d in data:
#                 d["_id"] = None
#         else:
#             result = table.insert_one(data)
#             data["_id"] = None
#         return result

#     def get_many(self, table, key, data):
#         try:
#             result = table.find({key: data}, {"_id": False})
#             result = list(result)
#             return result
#         except Exception:
#             return None

#     def get(self, table, key, data):
#         try:
#             result = table.find_one({key: data}, {"_id": False})
#             return result
#         except Exception:
#             return None

#     def get_all(self, table):
#         try:
#             result = table.find(
#                 {},
#                 {
#                     "_id": False,
#                 },
#             )
#             return list(result)
#         except Exception:
#             return None

#     def get_one(self, table, key, data):
#         try:
#             result = table.find_one({key: data}, {"_id": False})
#             return dict(result)
#         except Exception:
#             return None


# DB = Database()

# # all=DB.get_all(DB.posts)

# # for post in all[:1]:
# #     if "interests" not in post:
# #         lst=list()
# #         med_card=DB.get_one(DB.media_cards,'id',post['parent_id'])
# #         lst.extend(med_card['subcategory'])
# #         lst.extend(med_card['subtype'])
# #         DB.posts.find_one_and_update(
# #             {"id": post['id']}, {"$set": {"interests": lst}}
# #         )
# #         print(post['id'])
