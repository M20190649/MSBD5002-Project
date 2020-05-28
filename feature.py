import re
import pandas as pd
from datetime import datetime

# Pandas's display settings
pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


def load():
    dataset_train = pd.read_csv("data/phase1_training/20min_avg_travel_time_training_phase1.csv")
    dataset_test = pd.read_csv("data/submission_sample/submission_sample_travelTime.csv")

    # Process date
    dataset_train, dataset_test = process_time_window(dataset_train), process_time_window(dataset_test)

    # Load weather from csv files
    weather_train, weather_test = load_weather()

    # Process weather
    dataset_train, dataset_test = process_weather(dataset_train, weather_train), \
                                  process_weather(dataset_test, weather_test)

    # Load route from csv files
    dataset_route = load_route()

    # Process route
    dataset_train, dataset_test = process_route(dataset_train, dataset_route), \
                                  process_route(dataset_test, dataset_route)
    

    # Append
    split = len(dataset_train)
    dataset = pd.concat([dataset_train, dataset_test], ignore_index=True)
    
    # Remove National Day (10-1 ~ 10-7)
    # dataset = dataset[(dataset.date < '2016-10-01') | (dataset.date > '2016-10-07')]

    # One-hot
    dataset = pd.get_dummies(dataset, columns=['intersection_id', 'tollgate_id', 'interval', 'weekday'])

    # Drop
    dataset = dataset.drop(columns=['time_window', 'date', 'hour', 'link_seq'])

    # Min-max
    columns = ['pressure', 'sea_pressure', 'wind_direction', 'wind_speed', 'temperature', 'rel_humidity',
               'precipitation', 'length', 'area', 'lanes', 'in_top', 'out_top']
    for column in columns:
        dataset[column] = (dataset[column] - dataset[column].min()) / (dataset[column].max() - dataset[column].min())

    # Split
    return dataset.iloc[:split].reset_index(drop=True), dataset.iloc[split:].reset_index(drop=True)


def load_weather():
    weather_train = pd.read_csv("data/weather/weather_July_01_Oct_17_table7.csv")
    weather_test = pd.read_csv("data/weather/weather_Oct_18_Oct_24_table7.csv")

    return weather_train, weather_test


def load_route():
    dataset_route = pd.read_csv("data/road/routes_table 4.csv")
    dataset_link = pd.read_csv("data/road/links_table3.csv")

    link_ids = dataset_link['link_id'].unique()
    for link_id in link_ids:
        dataset_route['pass_%s' % link_id] = 0

    columns = ['length', 'area', 'lanes', 'in_top', 'out_top']
    for column in columns:
        dataset_route[column] = 0

    def transform_link(seq, links):
        seq = seq.split(',')
        links = links[links['link_id'].isin(seq)].copy()

        links['area'] = links['length'] * links['width']
        links['in_top_count'] = links['in_top'].apply(lambda x: 0 if pd.isna(x) else len(x.split(',')))
        links['out_top_count'] = links['out_top'].apply(lambda x: 0 if pd.isna(x) else len(x.split(',')))

        total_length = links['length'].sum()
        total_area = links['area'].sum()

        links['weighted_lanes'] = links['length'] * links['lanes'] / total_length
        links['weighted_in_top'] = links['area'] * links['in_top_count'] / total_area
        links['weighted_out_top'] = links['area'] * links['out_top_count'] / total_area

        return seq, total_length, total_area, \
               links['weighted_lanes'].sum(), links['weighted_in_top'].sum(), links['weighted_out_top'].sum()

    for index, row in dataset_route.iterrows():
        link_seq, length, area, lanes, in_top, out_top = transform_link(row['link_seq'], dataset_link)

        for link_id in link_seq:
            row['pass_%s' % link_id] = 1

        row['length'], row['area'], row['lanes'], row['in_top'], row['out_top'] = length, area, lanes, in_top, out_top
        dataset_route.iloc[index] = row

    return dataset_route


def process_time_window(dataset):
    def transform_date(time_window):
        pattern = re.compile(r'(\d{4})-(\d{2})-(\d{2}) (\d{2}):(\d{2})')
        time_window = pattern.findall(time_window)[0]
        return '%s-%s-%s' % (time_window[0], time_window[1], time_window[2])

    def transform_interval(time_window):
        pattern = re.compile(r'(\d{4})-(\d{2})-(\d{2}) (\d{2}):(\d{2})')
        time_window = pattern.findall(time_window)[0]
        return int(int(time_window[3]) * 3 + int(time_window[4]) / 20)

    def transform_hour(time_window):
        pattern = re.compile(r'(\d{4})-(\d{2})-(\d{2}) (\d{2}):(\d{2})')
        time_window = pattern.findall(time_window)[0]
        return int(time_window[3]) - int(time_window[3]) % 3

    def transform_weekday(date):
        return datetime.strptime(date, '%Y-%m-%d').weekday()

    dataset['date'] = dataset['time_window'].apply(lambda x: transform_date(x))
    dataset['interval'] = dataset['time_window'].apply(lambda x: transform_interval(x))
    dataset['hour'] = dataset['time_window'].apply(lambda x: transform_hour(x))
    dataset['weekday'] = dataset['date'].apply(lambda x: transform_weekday(x))

    return dataset


def process_weather(dataset, weather):
    return pd.merge(dataset, weather, on=['date', 'hour'], how='left')


def process_route(dataset, route):
    return pd.merge(dataset, route, on=['intersection_id', 'tollgate_id'], how='left')

