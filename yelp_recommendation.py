# Cameron Ela, ceela@usc.edu
# DSCI 553, Spring 2025
# Competition Project
"""
Method Description:
I used a model-based approach for this project. In my testing, I found that my validation set error only increased when
including collaborative filtering to have a hybrid approach. As I lowered the threshold for number of reviews needed to
use the XGBRegressor instead of collaborative filtering, RMSE continued to drop. The continuous drop in RMSE confirmed
my suspicion that my collaborative filtering algorithm was not helpful in this case, and a hybrid approach would not be
ideal. From homework 3 to this project, I added the 'elite' field from user.json as a feature since it legitimizes a
user's popularity and therefore, perhaps, is a good indicator of how a restaurant should be rated. Adding this field,
holding all else equal and before making the following changes, had the biggest impact on reducing RMSE. My original
XGBRegressor model submitted with homework 3 was built from cross validation on maximum tree depth, number of trees,
and column sample size per tree. This time, to aid the column sampling in preventing overfitting, I added an L1
regularization term to search for in cross-validation of hyperparameters for XGBRegressor. The addition of the L1
regularization parameter (alpha) in the final model had minor positive impact for RMSE. Additionally, I tried adding
number of likes on tips for both users and businesses as features but realized it is confounding with the number of
tips, making number of likes on tips unnecessary.

Error Distribution:
>=0 and <1: 102083
>=1 and <2: 33007
>=2 and <3: 6139
>=3 and <4: 815
>=4: 0

RMSE:
0.9792830344697815

Execution Time:
96s
"""

from pyspark import SparkContext
import sys
import numpy as np
import json
from xgboost import XGBRegressor
import time
# import math
# from sklearn.model_selection import GridSearchCV


def build_user_features(line, tips, text_review_count):
    line = json.loads(line)
    pair = tuple(
        (line['user_id'], (
            line['review_count'], 0 if line['friends'] == 'None' else len(line['friends']),
            line['useful'], line['funny'], line['cool'], line['fans'], line['average_stars'],
            0 if line['elite'] == 'None' else len(line['elite']),
            sum([line['compliment_hot'], line['compliment_more'], line['compliment_profile'], line['compliment_cute'],
                 line['compliment_list'], line['compliment_note'], line['compliment_plain'], line['compliment_cool'],
                 line['compliment_funny'], line['compliment_writer'], line['compliment_photos']]),
            tips.get(line['user_id'], 0), text_review_count.get(line['user_id'], 0)
        )
         )
    )
    return pair


def build_business_features(line, checkin_count, photo_count, tips, text_review_count):
    line = json.loads(line)
    features = (line['stars'], line['review_count'], line['is_open'],
                0 if line['categories'] is None else len(line['categories'].split(',')),
                *([0] * 19 if line['attributes'] is None else [
                    int(bool(line['attributes'].get('Alcohol', None))),
                    int(bool(line['attributes'].get('RestaurantsDelivery', None))),
                    int(bool(line['attributes'].get('GoodForKids', None))),
                    int(bool(line['attributes'].get('OutdoorSeating', None))),
                    int(bool(line['attributes'].get('RestaurantsGoodForGroups', None))),
                    int(bool(line['attributes'].get('RestaurantsTableService', None))),
                    int(bool(line['attributes'].get('RestaurantsTakeOut', None))),
                    int(bool(line['attributes'].get('Caters', None))),
                    int(bool(line['attributes'].get('WheelchairAccessible', None))),
                    line.get('RestaurantsPriceRange2', None)
                ]),
                )
    if len(features) == 14:
        features += tuple(
            [0] * 9 if not isinstance(line['attributes'].get('Ambience', None), dict) else [
                int(bool(line['attributes']['Ambience'].get('romantic', None))),
                int(bool(line['attributes']['Ambience'].get('intimate', None))),
                int(bool(line['attributes']['Ambience'].get('classy', None))),
                int(bool(line['attributes']['Ambience'].get('hipster', None))),
                int(bool(line['attributes']['Ambience'].get('divey', None))),
                int(bool(line['attributes']['Ambience'].get('touristy', None))),
                int(bool(line['attributes']['Ambience'].get('trendy', None))),
                int(bool(line['attributes']['Ambience'].get('upscale', None))),
                int(bool(line['attributes']['Ambience'].get('casual', None)))
            ]
        )
    features += (checkin_count.get(line['business_id'], 0), photo_count.get(line['business_id'], 0),
                 tips.get(line['business_id'], 0), text_review_count.get(line['business_id'], 0))

    return tuple((line['business_id'], features))


def get_user_business_vectors(context, folder):
    # read in text reviews and tips
    text_review_json = context.textFile(f'{folder}/review_train.json') \
        .map(lambda line: json.loads(line))
    tips_json = context.textFile(f'{folder}/tip.json') \
        .map(lambda line: json.loads(line))

    # build user info
    uid_text_review_count = text_review_json.map(lambda line: (line['user_id'], line['review_id'])) \
        .groupByKey() \
        .mapValues(lambda val: len(val)) \
        .collectAsMap()
    uid_tips = tips_json.map(lambda line: (line['user_id'], 1)) \
        .reduceByKey(lambda x, y: x + y) \
        .collectAsMap()
    uid_info = context.textFile(f'{folder}/user.json') \
        .map(lambda line: build_user_features(line, uid_tips, uid_text_review_count)) \
        .collectAsMap()
    del uid_text_review_count, uid_tips

    # build business info
    bid_checkin_count = context.textFile(f'{folder}/checkin.json') \
        .map(lambda line: json.loads(line)) \
        .map(lambda line: (line['business_id'], sum(line['time'].values()))) \
        .collectAsMap()
    bid_photo_count = context.textFile(f'{folder}/photo.json') \
        .map(lambda line: json.loads(line)) \
        .map(lambda line: (line['business_id'], line['photo_id'])) \
        .groupByKey() \
        .mapValues(lambda val: len(val)) \
        .collectAsMap()
    bid_tips = tips_json.map(lambda line: (line['business_id'], 1)) \
        .reduceByKey(lambda x, y: x + y) \
        .collectAsMap()
    bid_text_review_count = text_review_json.map(lambda line: (line['business_id'], line['review_id'])) \
        .groupByKey() \
        .mapValues(lambda val: len(val)) \
        .collectAsMap()
    bid_info = context.textFile(f'{folder}/business.json') \
        .map(
        lambda line: build_business_features(
            line, bid_checkin_count, bid_photo_count, bid_tips, bid_text_review_count
        )) \
        .collectAsMap()

    return uid_info, bid_info


def create_feature_vectors(partition, uid_info, bid_info):
    vectors = []
    for row in partition:
        user_features = uid_info.value[row[0]]
        business_features = bid_info.value[row[1]]
        if len(row) == 3:
            vectors.append(np.array((row[0], row[1]) + user_features + business_features + (row[2],)))
        else:
            vectors.append(np.array((row[0], row[1]) + user_features + business_features))
    return vectors


# def find_best_xgb_params(X_train, y_train):
#     parameters = {
#         'max_depth': [2, 3, 4, 5, 6],
#         'n_estimators': [100, 125, 150, 175, 200, 225, 250],
#         'booster': ['gbtree'],
#         'alpha': [1, 10, 100, 1000],
#         'colsample_bytree': [0.75, 0.8, 0.85, 0.9],
#         'random_state': [2025]
#     }
#     gscv = GridSearchCV(XGBRegressor(), param_grid=parameters, scoring='neg_mean_squared_error', verbose=2, n_jobs=-1)
#     gscv.fit(X_train, y_train)
#     best_params = gscv.best_params_
#
#     return best_params


def main(folder_path, test_file_name, output_file_name):
    # initialize SparkContext
    start = time.time()
    sc = SparkContext(master='local[*]', appName='competition')
    uid_info, bid_info = get_user_business_vectors(sc, folder_path)
    uid_info_bc, bid_info_bc = sc.broadcast(uid_info), sc.broadcast(bid_info)

    # create training and validation datasets
    # load training data
    train_file = sc.textFile(f'{folder_path}/yelp_train.csv') \
        .map(lambda line: line.split(','))
    header_1 = train_file.first()
    train_file_parsed = train_file.filter(lambda line: line != header_1) \
        .map(lambda line: (line[0], line[1], float(line[2])))
    train_data = np.array(
        train_file_parsed.mapPartitions(
            lambda partition: create_feature_vectors(partition, uid_info_bc, bid_info_bc)).collect()
    )
    # load test data
    test_file = sc.textFile(test_file_name) \
        .map(lambda line: line.split(','))
    header_2 = test_file.first()
    test_file_parsed = test_file.filter(lambda line: line != header_2) \
        .map(lambda line: (line[0], line[1]))
    test_data = np.array(
        test_file_parsed.mapPartitions(
            lambda partition: create_feature_vectors(partition, uid_info_bc, bid_info_bc)).collect()
    )

    X_train = train_data[:, 2:-1]
    y_train = train_data[:, -1]
    X_test = test_data[:, 2:]

    # best_params = find_best_xgb_params(X_train, y_train)
    # print(best_params)

    # train and test XGBRegressor with model params
    # {'alpha': 100, 'booster': 'gbtree', 'colsample_bytree': 0.9, 'max_depth': 6, 'n_estimators': 200, 'random_state': 2025}
    xgb = XGBRegressor(
        booster='gbtree',
        colsample_bytree=0.9,
        max_depth=6,
        n_estimators=200,
        alpha=100,
        random_state=2025
    )
    xgb.fit(X_train, y_train)
    y_pred = xgb.predict(X_test)
    predictions = np.column_stack((test_data[:, 0:2], y_pred))

    # write out predictions in csv
    with open(output_file_name, 'w') as file:
        file.write('user_id,business_id,prediction\n')
        for row in predictions:
            file.write(f'{row[0]},{row[1]},{row[2]}\n')

    # # calculate duration, error distribution and RMSE
    # print('duration:', time.time() - start)
    #
    # output_file = sc.textFile(output_file_name)
    # header_3 = output_file.first()
    # output_file_parsed = output_file.filter(lambda pair: pair != header_3) \
    #     .map(lambda line: line.split(',')) \
    #     .map(lambda row: ((row[0], row[1]), float(row[2])))
    # test_file_with_val = test_file.filter(lambda pair: pair != header_2) \
    #     .map(lambda row: ((row[0], row[1]), float(row[2])))
    #
    # # error distribution
    # abs_errs = output_file_parsed.join(test_file_with_val) \
    #     .map(lambda row: abs(row[1][0] - row[1][1]))
    # err_dist = abs_errs.map(lambda err: (
    #     '>=0 and <1' if err < 1 else
    #     '>=1 and <2' if err < 2 else
    #     '>=2 and <3' if err < 3 else
    #     '>=3 and <4' if err < 4 else
    #     '>=4'
    # )).countByValue()
    # print(err_dist)
    #
    # # RMSE
    # squared_errs = abs_errs.map(lambda val: val ** 2)
    # rmse = math.sqrt(squared_errs.mean())
    # print('RMSE:', rmse)

    sc.stop()


if __name__ == '__main__':
    if len(sys.argv) != 4:
        print('CLI Args: competition.py <folder_path> <test_file_name> <output_file_name>')
        sys.exit(1)

    folder_path = sys.argv[1]
    test_file_name = sys.argv[2]
    output_file_name = sys.argv[3]
    main(folder_path, test_file_name, output_file_name)
