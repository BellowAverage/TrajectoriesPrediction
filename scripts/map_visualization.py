import os
import folium
import pandas as pd


def map_matching(data):
    demo_map = folium.Map(
        tiles='https://wprd01.is.autonavi.com/appmaptile?x={x}&y={y}&z={z}&lang=zh_cn&size=1&scl=1&style=7',
        attr='高德-常规图',
        location=[39.93, 116.40],
        zoom_start=12)

    locations = []
    for latitude, longitude in zip(data['Latitude'], data['Longitude']):
        locations.append([latitude, longitude])
    print(locations)

    folium.PolyLine(
        locations=locations,
        weight=3,
        color='red',
        opacity=0.8
    ).add_to(demo_map)

    demo_map.save('demo_map.html')
    os.system(r"demo_map.html")


def map_matching_plot(data, file_name):
    demo_map = folium.Map(
        tiles='https://wprd01.is.autonavi.com/appmaptile?x={x}&y={y}&z={z}&lang=zh_cn&size=1&scl=1&style=7',
        attr='高德-常规图',
        location=[39.93, 116.40],
        zoom_start=12)

    for i in range(data.shape[0]):
        tmp = data.iloc[i]
        pop_content = 'Altitude: ' + str(tmp['Altitude']) + "<br>" + 'Date: ' + str(tmp['Date'])

        folium.CircleMarker(
            location=(tmp['Latitude'], tmp['Longitude']),
            radius=2,
            popup=folium.Popup(pop_content, max_width=2000),
            color='red',
            fill=True,
            fill_color='red',
        ).add_to(demo_map)

    demo_map.save(file_name)
    os.system(file_name)