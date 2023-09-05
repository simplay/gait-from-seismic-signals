import datetime
import os
from os import listdir
from os.path import isfile, join
import logging

import numpy as np
import pandas as pd
import scipy as sp

PEAK_DIST = 50  # for normal walks
PEAK_DIST = 40  # short fast

TASKS = {
    "bf8a163b-8f55-4885-b3ab-cf8d26f3904c": "00 Free Walk",
    "4b53b220-1c96-4af5-a3a5-0d377fd22b2a": "01 Heel Walk",
    "f4cc5bf3-c8a3-448a-832f-6dfe33a1cf71": "02 Forefoot Walk",
    "3acbec5c-f0a4-4025-9d48-9bf139175f54": "03 Fast Walk Short",
    "921f551e-fdfe-4760-8942-94d6882d3493": "04 Normal Walk Short",
    "6aed9ef9-6379-483a-a222-1e1c63b23bd0": "05 Slow Walk Short",
    "65142a2d-b26d-45a0-906e-89d63fca32e5": "06 Fast Walk Long",
    "f32593da-4168-4684-8653-567186458004": "07 Normal Walk Long",
    "66aae626-9f67-4a97-94a2-80ddec1a030b": "08 Slow Walk Long",
    "3689a22b-1e73-4086-8d36-498bf6ba4376": "09 Wide Short Step Walk",
    "ace69e32-3d65-4d37-aaa6-3b693141ec45": "10 Narrow Short Step Walk",
    "84cfb85c-14db-4740-8cc2-6711863a54e2": "11 Stiff Leg Walk",
    "853840fc-1b11-4e31-882f-24a5456862ea": "12 Calculation Task Walk",
    "c6c44aa1-bcfc-4f8d-a451-e42dac3c6a2b": "13 FGA 01",
    "884605b7-d5b2-450d-89c8-70d253ce9227": "14 FGA 02",
    "4d3767fb-e7fd-4fd4-988e-bec0809bbfc5": "15 FGA 03",
    "0bc7a3cf-ad18-4aa1-8f04-f59e809188d4": "16 FGA 04",
    "3acb8f9e-543b-47cd-8bde-b3f235f43a7c": "17 FGA 05",
    "555bf468-956a-44f9-9b1c-91783b546096": "18 FGA 06",
    "0057556d-e73e-4d7d-ba71-0cbbe5f884b5": "19 FGA 07",
    "6fcb5b58-e620-470e-8a4a-03f8bbcec368": "20 FGA 08",
    "906f3b35-af7f-44cb-8e21-f7d7b7a1f523": "21 FGA 09",
    "8f431666-605f-4864-936e-daa93dcbe3df": "22 FGA 10",
    "487353b5-70e6-4083-a70e-756bc3d290df": "23 5XSST",
    "84350f1e-4525-4198-87c2-0e92a04b66ab": "24 TUG",
    "e9df7bef-2085-436d-b3dd-e568057b83ee": "25 Free Walk 01",
    "1807930d-965a-4ffe-b7ac-4bae9dafadbb": "26 Free Walk 02",
    "ea9e5a78-6528-4bc5-b07d-190bd35ed32f": "27 Free Walk 03",
    "0a039c05-b502-4a92-904f-4b2905bfd502": "28 Free Walk 04",
    "8aafb648-0e30-4c48-889b-0f98080d479c": "29 Free Walk 05",
    "d300eb92-72e7-40a0-bbe3-1b3d79627d09": "30 Free Walk 06",
    "01f18b93-1128-4b9a-8bee-285886fe0c81": "31 ADL Tie Shoes",
    "ebcfad23-1a53-45a8-96cf-5952d6295138": "32 ADL Couch Sit Down",
    "9544f797-d115-4f7d-b6ae-5358c4142b04": "33 ADL Dining",
    "6d0f2757-17bf-4749-adee-36e37a66326e": "34 ADL Put Dishes",
    "c94476a5-d286-4c10-baee-49fcddb7f4c3": "35 ADL Item Drop",
    "74b94256-e4d5-451e-b734-d84ab077cd10": "NONE",
}

PARTICIPANT_UUIDS_TO_SHORTNAME = {
    "5772ae35-0339-4c2b-81ec-9404cd507d5d": "p01",
    "9e3bb078-c0f2-49af-b4b2-f3e60c9acbd9": "p02",
    "4563f65d-7bba-4117-b26d-17bc59880da3": "p03",
    "e7fe586a-4f30-4d78-a5df-d0f894486e7c": "p04",
    "7d4fbfca-682f-4620-99bb-8e78e8db8dcd": "p05",
    "65097f7c-7afb-4635-bb15-a5ed007ae1a9": "p06",
    "c4de5267-49d4-49b9-901d-5f198adab210": "p07",
    "f9853d6f-7a10-4f12-adb8-5f853eae9b00": "p08",
    "ad6475b0-60fb-4085-afcb-7b81635af562": "p09",
    "94905937-68ed-46e9-a69b-2be4794d1677": "p10",
    "b43bceaa-6df2-4263-b99d-4886ada4bc51": "p11",
    "849fa969-81bd-4176-b427-734cde6b7247": "p12",
    "9c323b0b-8542-46b7-b942-771c3ac93b92": "p13",
    "b519af76-c305-4753-8a20-e881aaf4290d": "p14",
    "15884390-24bc-4b68-8ad0-315e4d2e3341": "p15",
    "48e8f2b5-9384-4fa0-b31f-a7f82cf8227d": "p16",
    "7bcf7a8d-8194-4595-87db-637d633ae776": "p17",
    "d4303bf5-1c55-4275-a7cc-4682a15df453": "p18",
    "7eb64377-58d0-47ad-bba2-a5e829386d86": "p19",
    "ca6be1af-b3e7-4f93-97d7-34bee5f37512": "p20",
    "30aee593-e682-4d09-b0ad-fb310565cb67": "p21",
    "7185048d-8ef9-4feb-a5c1-81880d6ff457": "p22",
    "06931492-57c2-4bb4-a480-690beb7a6d79": "p23",
    "04ed494e-9b92-46bd-992d-935212cd5182": "p24",
    "efab3d8a-ebee-494a-acce-d1e85f751e0b": "p25",
    "0cafb5fc-8622-476b-9ab8-106e8996a564": "p26",
    "780c61e5-3e93-40de-8f95-22d0c4f49933": "p27",
    "34dd12d0-c536-420c-885d-72759c9d3e34": "p28",
    "d46f44a0-1ac4-4329-8396-b22e4c9f2c8d": "p29",
    "52051846-b274-424e-bb5d-9997c257cc14": "p30",
    "dd2baa97-1311-4e51-a04f-b266b0b14b45": "p31",
    "80f1ebe2-cb1a-4c77-b4fd-343796a5cc99": "p32",
    "43de6126-73f9-429c-8774-9af05358e426": "p33",
    "7b5f5199-8495-41ac-8fe1-1b5c92906050": "p34",
    "60d94d65-775e-4a24-acf3-8513e511c797": "p35",
    "b5189c9a-e26a-4be5-bcb8-294811ae5b97": "p36",
    "2735411b-a398-41eb-acd1-3a6cad77e793": "p37",
    "8d7ea892-8aba-4a7b-9149-82882939686c": "p38",
    "96b41679-aabd-4914-85d1-7aec4f8b6ed0": "p39",
    "538f0f89-cde5-4b4c-855f-27013bc0e221": "p40",
    "19ee1db4-7530-40bb-ad69-f18267cf891c": "p41",
    "c3445518-be61-49d0-9cc2-8ce6b538a501": "p42",
    "156a58f6-024f-4fff-b524-c442258ef2b7": "p43",
    "5a3f65a8-8988-4c0b-99ca-8bdbf826c7c9": "p44",
    "341793e1-e14b-42b5-831d-bc37c9120a8a": "p45",
    "687de875-ecef-4828-9ada-e1b93243a5ff": "p46",
    "87ad7dd4-ab5d-4870-a148-b89d3d6da4b6": "p47",
    "cd490cca-ab46-4fcf-a6bc-5005ef5b3c08": "p48",
    "ab09248f-1b28-4b00-b7a4-ca27c4b70066": "p49",
    "23bdba63-c339-4ad8-9d75-0d0cca2bd2d5": "p50",
    "bb2894e4-c4ac-454e-b559-b03370cc6f39": "p51",
    "3231001d-765f-4312-8eba-016ae8ece18c": "p52",
    "5aae387e-e184-4709-9786-036ac2a4c8aa": "p53",
    "fcc96631-2af8-4fb2-b9f5-d69420deed9a": "p54",
    "68786383-09ae-4aae-89ee-d57b0cedda1b": "p55",
    "efa9412e-ce02-4748-b31f-c33d352390e2": "p56",
    "5c8d2602-02a6-4858-82b1-47f0731a6a28": "p57",
    "193372ae-5356-4415-906b-e07b7b1d955d": "p58",
    "27c465bd-d509-4a6d-ac3c-afbcdf63a5a7": "p59",
    "7b546f5d-113b-4d1c-8bdd-d7d320a429a8": "p60",
    "a3e27435-0be3-4237-b738-5b06e9bc823b": "p61",
    "5eac1a0b-a859-42d6-889d-aa1489b4897c": "p67",
    "1b1fc3dd-0687-416d-9523-4e66bfa1a1aa": "p62",
    "5ca9971f-2486-4b02-80f7-6c1f854dc7df": "p63",
    "d8f3f787-705e-4884-80c4-bd9e8274c613": "p64",
    "5101fe34-5d80-4396-9280-fd70638309e1": "p65",
    "eddb03a8-b539-425f-96fc-9474957df22e": "p66",
}

SAMPLE_NAME_TO_PARTICIPANT_UUID = {
    "parquet_samples_24_03_22_1648122763948": "11e95935-a17a-4469-a453-9215a3e00ae2",
    "parquet_samples_24_03_22_1648122853865": "d8f3f787-705e-4884-80c4-bd9e8274c613",
    "parquet_samples_24_03_22_1648123186720": "1b1fc3dd-0687-416d-9523-4e66bfa1a1aa",
    "parquet_samples_24_03_22_1648123228495": "b5189c9a-e26a-4be5-bcb8-294811ae5b97",
    "parquet_samples_24_03_22_1648123336006": "5aae387e-e184-4709-9786-036ac2a4c8aa",
    "parquet_samples_24_03_22_1648123393912": "5eac1a0b-a859-42d6-889d-aa1489b4897c",
    "parquet_samples_24_03_22_1648123442442": "23bdba63-c339-4ad8-9d75-0d0cca2bd2d5",
    "parquet_samples_24_03_22_1648123490467": "5101fe34-5d80-4396-9280-fd70638309e1",
    "parquet_samples_24_03_22_1648123602777": "eddb03a8-b539-425f-96fc-9474957df22e",
    "parquet_samples_24_03_22_1648123745523": "7d4fbfca-682f-4620-99bb-8e78e8db8dcd",
    "parquet_samples_24_03_22_1648123791040": "60d94d65-775e-4a24-acf3-8513e511c797",
    "parquet_samples_24_03_22_1648123860726": "4563f65d-7bba-4117-b26d-17bc59880da3",
    "parquet_samples_24_03_22_1648123940540": "7185048d-8ef9-4feb-a5c1-81880d6ff457",
    "parquet_samples_24_03_22_1648124007355": "52051846-b274-424e-bb5d-9997c257cc14",
    "parquet_samples_24_03_22_1648124127705": "7eb64377-58d0-47ad-bba2-a5e829386d86",
    "parquet_samples_24_03_22_1648124177503": "30aee593-e682-4d09-b0ad-fb310565cb67",
    "parquet_samples_24_03_22_1648124237843": "5772ae35-0339-4c2b-81ec-9404cd507d5d",
    "parquet_samples_24_03_22_1648124298906": "341793e1-e14b-42b5-831d-bc37c9120a8a",
    "parquet_samples_24_03_22_1648124348640": "15884390-24bc-4b68-8ad0-315e4d2e3341",
    "parquet_samples_24_03_22_1648124394090": "94905937-68ed-46e9-a69b-2be4794d1677",
    "parquet_samples_24_03_22_1648124548194": "5a3f65a8-8988-4c0b-99ca-8bdbf826c7c9",
    "parquet_samples_24_03_22_1648124588873": "9e3bb078-c0f2-49af-b4b2-f3e60c9acbd9",
    "parquet_samples_24_03_22_1648124635447": "80f1ebe2-cb1a-4c77-b4fd-343796a5cc99",
    "parquet_samples_24_03_22_1648124692652": "65097f7c-7afb-4635-bb15-a5ed007ae1a9",
    "parquet_samples_24_03_22_1648124750281": "c4de5267-49d4-49b9-901d-5f198adab210",
    "parquet_samples_24_03_22_1648124789542": "d46f44a0-1ac4-4329-8396-b22e4c9f2c8d",
    "parquet_samples_24_03_22_1648124871361": "156a58f6-024f-4fff-b524-c442258ef2b7",
    "parquet_samples_24_03_22_1648124955630": "65097f7c-7afb-4635-bb15-a5ed007ae1a9",
    "parquet_samples_24_03_22_1648124998316": "bb2894e4-c4ac-454e-b559-b03370cc6f39",
    "parquet_samples_24_03_22_1648125044929": "ca6be1af-b3e7-4f93-97d7-34bee5f37512",
    "parquet_samples_24_03_22_1648125082963": "dd2baa97-1311-4e51-a04f-b266b0b14b45",
    "parquet_samples_24_03_22_1648125135863": "f9853d6f-7a10-4f12-adb8-5f853eae9b00",
    "parquet_samples_24_03_22_1648125660341": "9c323b0b-8542-46b7-b942-771c3ac93b92",
    "parquet_samples_24_03_22_1648125706588": "43de6126-73f9-429c-8774-9af05358e426",
    "parquet_samples_24_03_22_1648125744345": "48e8f2b5-9384-4fa0-b31f-a7f82cf8227d",
    "parquet_samples_24_03_22_1648125781082": "ad6475b0-60fb-4085-afcb-7b81635af562",
    "parquet_samples_24_03_22_1648125813947": "e7fe586a-4f30-4d78-a5df-d0f894486e7c",
    "parquet_samples_24_03_22_1648125847726": "efab3d8a-ebee-494a-acce-d1e85f751e0b",
    "parquet_samples_24_03_22_1648125935030": "34dd12d0-c536-420c-885d-72759c9d3e34",
    "parquet_samples_24_03_22_1648125979847": "780c61e5-3e93-40de-8f95-22d0c4f49933",
    "parquet_samples_24_03_22_1648126030771": "2735411b-a398-41eb-acd1-3a6cad77e793",
    "parquet_samples_24_03_22_1648126061391": "b519af76-c305-4753-8a20-e881aaf4290d",
    "parquet_samples_24_03_22_1648126094385": "d4303bf5-1c55-4275-a7cc-4682a15df453",
    "parquet_samples_24_03_22_1648126307770": "04ed494e-9b92-46bd-992d-935212cd5182",
    "parquet_samples_24_03_22_1648126343454": "19ee1db4-7530-40bb-ad69-f18267cf891c",
    "parquet_samples_24_03_22_1648126373412": "87ad7dd4-ab5d-4870-a148-b89d3d6da4b6",
    "parquet_samples_24_03_22_1648126401139": "96b41679-aabd-4914-85d1-7aec4f8b6ed0",
    "parquet_samples_24_03_22_1648126429511": "cd490cca-ab46-4fcf-a6bc-5005ef5b3c08",
    "parquet_samples_24_03_22_1648126655693": "7b5f5199-8495-41ac-8fe1-1b5c92906050",
    "parquet_samples_24_03_22_1648126698417": "687de875-ecef-4828-9ada-e1b93243a5ff",
    "parquet_samples_24_03_22_1648126830105": "5ca9971f-2486-4b02-80f7-6c1f854dc7df",
    "parquet_samples_24_03_22_1648126878118": "7bcf7a8d-8194-4595-87db-637d633ae776",
    "parquet_samples_24_03_22_1648126915755": "3231001d-765f-4312-8eba-016ae8ece18c",
    "parquet_samples_24_03_2_1648126948582": "a3e27435-0be3-4237-b738-5b06e9bc823b",
    "parquet_samples_24_03_22_1648126985419": "ab09248f-1b28-4b00-b7a4-ca27c4b70066",
    "parquet_samples_24_03_22_1648127105545": "11e95935-a17a-4469-a453-9215a3e00ae2",
    "parquet_samples_24_03_22_1648127172174": "b43bceaa-6df2-4263-b99d-4886ada4bc51",
    "parquet_samples_24_03_22_1648127250996": "7b546f5d-113b-4d1c-8bdd-d7d320a429a8",
    "parquet_samples_24_03_22_1648127326779": "8d7ea892-8aba-4a7b-9149-82882939686c",
    "parquet_samples_24_03_22_1648127356644": "27c465bd-d509-4a6d-ac3c-afbcdf63a5a7",
    "parquet_samples_24_03_22_1648127413003": "193372ae-5356-4415-906b-e07b7b1d955d",
    "parquet_samples_24_03_22_1648127441824": "68786383-09ae-4aae-89ee-d57b0cedda1b",
    "parquet_samples_24_03_22_1648127465345": "efa9412e-ce02-4748-b31f-c33d352390e2",
    "parquet_samples_24_03_22_1648127529045": "0cafb5fc-8622-476b-9ab8-106e8996a564",
    "parquet_samples_24_03_22_1648127567997": "c3445518-be61-49d0-9cc2-8ce6b538a501",
    "parquet_samples_24_03_22_16481276405732": "fcc96631-2af8-4fb2-b9f5-d69420deed9a",
}


def structural_element(radius):
    se = np.zeros(radius * 2 + 1)
    for k in range(len(se)):
        se[k] = np.sqrt((radius ** 2) - (radius - k) ** 2)
    return se


def erosion(data, se):
    width = len(se)
    radius = (width - 1) / 2

    data = np.insert(data, 0, data[0].repeat(radius), 0)
    data = np.append(data, data[-1].repeat(radius))
    mask = np.zeros(len(data) - width + 1)

    for k in range(len(data) - width + 1):
        mask[k] = min(data[k:k + width] - se)

    return mask


def dilation(data, se):
    width = len(se)
    radius = (width - 1) / 2

    data = np.insert(data, 0, data[0].repeat(radius), 0)
    data = np.append(data, data[-1].repeat(radius))
    mask = np.zeros(len(data) - width + 1)

    for k in range(len(data) - len(se) + 1):
        mask[k] = max(data[k:k + width] + se)

    return mask


def opening(data, se):
    data = erosion(data, se)
    data = dilation(data, se)
    return data


def closing(data, se):
    data = dilation(data, se)
    data = erosion(data, se)
    return data


def load_seismograph_parquets(parquet_filepaths: list):
    dataframes = {}
    ip_list = ["156", "48", "93"]
    for idx, parquet_filepath in enumerate(parquet_filepaths):
        seismo_filepath = parquet_filepath  # os.path.join(base_seismograph_path, parquet_filepath)
        logging.info(seismo_filepath)
        dataframes[ip_list[idx]] = pd.read_parquet(seismo_filepath, engine="fastparquet")

    return dataframes


def extract_data(sample, proper_sensor_ip, task_id, trial_number):
    logging.info(f"{sample} {proper_sensor_ip} {task_id} {trial_number}")


def create_flattened_dataframes(df, target_channel="EHZ"):
    df_channel = df[df.channel_name == target_channel]

    first_timestamp = df_channel.timestamp.iloc[0]
    n = len(df_channel) * len(df_channel.iloc[0].measurement)

    # 100 Hz = 1000 ms / 100 sps = 10 ms
    dt = [first_timestamp + pd.Timedelta(10 * x, 'ms') for x in range(n)]
    measurements = [item for sublist in df_channel.measurement for item in sublist]
    
    '''
    if len(dt) != len(measurements):
        logging.info(
            f"Could not create flat file representation len(dt) = {len(dt)} and len(measurements) = {len(measurements)}")
        return None, False
    '''

    _df = pd.DataFrame(data={'dt': dt, 'data': measurements})
    return _df, True


def generate_segmented_parquet_files(dataframes, participant_id):
    sample = None
    for task_id, task_name in TASKS.items():
        for sensor_ip, df in dataframes.items():
            proper_sensor_ip = f"192.168.47.{sensor_ip}"
            trial_numbers_of_task = df[df["task_uuid"] == task_id]["trial_number"].unique()
            for trial_number in trial_numbers_of_task:
                sample = df[(df["task_uuid"] == task_id) & (df["trial_number"] == trial_number)]
                filename = f"{participant_id}_{proper_sensor_ip}_{task_id}_{trial_number}.parquet"
                output_filepath = os.path.join("extracted-trials-per-participant", filename)
                sample.to_parquet(output_filepath)
                logging.info(f"Wrote file {output_filepath}")

                # extract flat files
                flattened_df, ok = create_flattened_dataframes(sample)
                if ok:
                    flattened_filename = f"{participant_id}_{proper_sensor_ip}_{task_id}_{trial_number}_flat.parquet"
                    flattened_output_filepath = os.path.join("extracted-trials-per-participant",
                                                             flattened_filename)
                    flattened_df.to_parquet(flattened_output_filepath)
                    logging.info(f"Wrote file {flattened_output_filepath}")
                else:
                    flattened_filename = f"{participant_id}_{proper_sensor_ip}_{task_id}_{trial_number}_flat.parquet"
                    flattened_output_filepath = os.path.join("extracted-trials-per-participant",
                                                             flattened_filename)
                    logging.info(f"File not exported: {flattened_output_filepath}")


def extract_task_dataframes_for(task_filenames):
    task_dataframes = {}
    for task_filename in task_filenames:
        dataframe_filepath = os.path.join("extracted-trials-per-participant", task_filename)
        df = pd.read_parquet(dataframe_filepath)
        # TODO: either refactor the dt column usages or comment this out df = df.set_index('dt')
        task_dataframes[task_filename] = df
    return task_dataframes


def find_index_of_first_interesting_peak(df):
    data = np.array(df.data)
    data = (data - np.mean(data)) / (np.max(data) - np.min(data))  # * 10_000
    se = structural_element(radius=2)
    data_opening = opening(data, se)
    data_closing = closing(data, se)
    avg_data = (data_opening + data_closing) / 2
    residual = data - avg_data
    height_of_top_samples = np.quantile(residual, q=0.98)
    peaks, _ = sp.signal.find_peaks(residual, height=height_of_top_samples, distance=PEAK_DIST)

    # plt.plot(df.dt, residual, alpha=0.4, label="res")
    # plt.plot(df.dt[peaks[2]], residual[peaks[2]], "x", alpha=0.4,label="peak")
    # plt.legend()
    # plt.show()

    return peaks[1]


# 1 step is very small in the environment of a seismograph
# --> the timedifference between two heel contacts is about the same for each seismo.
# but if you look at the real situation and how the seismos are placed,
# the distance from heel contact to each seismo is important and you can see that the same
# step arrives at different times at the different seismos
def extract_activities(df, radius=11, height_quantile=0.98):
    data = np.array(df.data)
    data = (data - np.mean(data)) / (np.max(data) - np.min(data))  # * 10_000

    se = structural_element(radius=radius)
    data_opening = opening(data, se)
    data_closing = closing(data, se)

    avg_data = (data_opening + data_closing) / 2
    residual = data - avg_data

    height_of_top_samples = np.quantile(residual, q=height_quantile)
    peaks, _ = sp.signal.find_peaks(residual, height=height_of_top_samples, distance=PEAK_DIST)
    valley, _ = sp.signal.find_peaks(-residual, height=height_of_top_samples, distance=PEAK_DIST)

    # generate peak mask
    sequence = np.diff(peaks) < 80
    s = np.append(sequence, False)
    r = np.flip(np.append(np.flip(sequence), False))
    mask = np.logical_or(r, s)

    masked_peaks = peaks[mask]
    
    # print(find_index_of_first_interesting_peak(df))
    
    timestamp_first_interesting_peak = df.dt[find_index_of_first_interesting_peak(df)]
    masked_peaks = masked_peaks[
        df.dt[masked_peaks] > (timestamp_first_interesting_peak - datetime.timedelta(seconds=2))]

    return residual, masked_peaks


def filtered_step_times(step_times):
    filtered_step_times = []
    for step_time in step_times:
        if step_time < 1000:
            filtered_step_times.append(float(step_time))
        else:
            return filtered_step_times
    return filtered_step_times


def generate_trial_files(seismo_filepaths):
    for seismo_filepath in seismo_filepaths[1:]:
        try:
            part_uuid = SAMPLE_NAME_TO_PARTICIPANT_UUID[seismo_filepath.split("/")[-1]]
            part_short_name = PARTICIPANT_UUIDS_TO_SHORTNAME[part_uuid]
            seismo_parquet_filepaths = [os.path.join(seismo_filepath, f) for f in listdir(seismo_filepath) if
                                        isfile(join(seismo_filepath, f))]

            dataframes = load_seismograph_parquets(seismo_parquet_filepaths)
            generate_segmented_parquet_files(dataframes, participant_id=part_short_name)
            logging.info(f"Completed exporting files for {part_short_name}.")
        except Exception as err:
            logging.info(f"part_uuid = {part_uuid}")
            logging.info(err)


def normalize(data):
    d = data.copy()
    d = (d - np.min(d)) / (np.max(d) - np.min(d))
    d = d - np.mean(d)
    return d


def compute_statistics(trail_df, residuals):
    cdf = trail_df.copy()
    cres = residuals.copy()

    df_combined = cdf.copy()
    df_combined["data"] = df_combined["data"] ** 2
    res = cres ** 2

    win = sp.signal.windows.kaiser(200, beta=14)

    filtered_res = sp.signal.convolve(res, win, mode='same') / sum(win)

    win = sp.signal.windows.boxcar(1000)
    filtered_res = sp.signal.convolve(filtered_res, win, mode='same')
    filtered_res = filtered_res
    filtered_res = (filtered_res - np.min(filtered_res)) / (np.max(filtered_res) - np.min(filtered_res))
    mask = filtered_res > 0.6

    ncres = normalize(cres) * mask
    ncres = ncres * (ncres >= 0.1)
    height_of_top_samples = np.quantile(ncres, q=0.99)
    ncrespeaks, _ = sp.signal.find_peaks(normalize(ncres), height=height_of_top_samples, distance=PEAK_DIST)
    st = np.diff(cdf.dt[ncrespeaks]) / 1_000_000

    step_times = []
    for step_time in st:
        if float(step_time) > 1000:
            break

        step_times.append(float(step_time))

    step_times = [step_time for step_time in step_times if step_time < 900]
    logging.info(f"{step_times} => {len(step_times)} steps with {np.sum(step_times)} ms")


def load_flat_dataframes(parquet_filepath):
    flat_free_walk_task_filenames = [f for f in listdir(parquet_filepath) if isfile(join(parquet_filepath, f))]
    flat_free_walk_task_filenames = [f for f in flat_free_walk_task_filenames if "_flat" in f]
    return extract_task_dataframes_for(flat_free_walk_task_filenames)


def extract_features_from_df(
    dfs,  
    range_to_analyze=None, 
    se_radius=11, 
    height_quantile=0.98):

    df1, df2, df3 = dfs

    if range_to_analyze:
        start_date = pd.to_datetime(range_to_analyze[0])
        end_date = pd.to_datetime(range_to_analyze[1])
        
        mask1 = (df1['dt'] > start_date) & (df1['dt'] <= end_date)
        mask2 = (df2['dt'] > start_date) & (df2['dt'] <= end_date)
        mask3 = (df3['dt'] > start_date) & (df3['dt'] <= end_date)
        
        df1 = df1[mask1]
        df2 = df2[mask2]
        df3 = df3[mask3]
        
        df1 = df1.reset_index()
        df2 = df2.reset_index()
        df3 = df3.reset_index()
        
    res1, peaks1 = extract_activities(df1, se_radius, height_quantile)
    res2, peaks2 = extract_activities(df2, se_radius, height_quantile)
    res3, peaks3 = extract_activities(df3, se_radius, height_quantile)
    
    df_avg = pd.merge_asof(df1, df2, on='dt')
    df_avg = pd.merge_asof(df_avg, df3, on='dt')
    data_avg = {'data': df_avg[["data_x", "data_y", "data"]].mean(axis=1), 'dt': df_avg.dt}
    df_avg = pd.DataFrame(data=data_avg)

    df_med = pd.merge_asof(df1, df2, on='dt')
    df_med = pd.merge_asof(df_med, df3, on='dt')
    data_med = {'data': df_med[["data_x", "data_y", "data"]].median(axis=1), 'dt': df_med.dt}
    df_med = pd.DataFrame(data=data_med)
    
    df_max = pd.merge_asof(df1, df2, on='dt')
    df_max = pd.merge_asof(df_max, df3, on='dt')
    data_max = {'data': df_max[["data_x", "data_y", "data"]].max(axis=1), 'dt': df_max.dt}
    df_max = pd.DataFrame(data=data_max)
     
    #df_max['data3'] = df_avg.iloc[:, 1:4].max(axis=1)
    
    #df_avg = df_avg.drop(["data_x", "data_y", "data"], axis=1)
    #df_avg = df_avg.rename(columns={"data3": "data"})
    
    #df_avg = df_avg.drop(["data_x", "data_y", "data"], axis=1)
    #df_avg = df_avg.rename(columns={"data3": "data"})
    
    res_avg, peaks_avg = extract_activities(df_avg, se_radius, height_quantile)
    res_med, peaks_med = extract_activities(df_med, se_radius, height_quantile)
    res_max, peaks_max = extract_activities(df_max, se_radius, height_quantile)
    
    return {
        "df1": [df1, res1, peaks1],
        "df2": [df2, res2, peaks2],
        "df3": [df3, res3, peaks3],
        "avg": [df_avg, res_avg, peaks_avg],
        "med": [df_med, res_med, peaks_med],
        "max": [df_max, res_max, peaks_max],
    }


def extract_features(task_dataframes, 
                     participant_id, 
                     walk_uuid, trial_nr, 
                     range_to_analyze=None, 
                     se_radius=11, 
                     height_quantile=0.98):
    """
        @param range_to_analyze None or array with two string elements that encode datetime objects (the left-and right part bewtween a certain time range. Example: range_to_analyze = ["09:15", "09:20"]
    """
    fn1 = f"{participant_id}_192.168.47.156_{walk_uuid}_{trial_nr}_flat.parquet"
    fn2 = f"{participant_id}_192.168.47.48_{walk_uuid}_{trial_nr}_flat.parquet"
    fn3 = f"{participant_id}_192.168.47.93_{walk_uuid}_{trial_nr}_flat.parquet"

    logging.info(f"{participant_id} - {TASKS[walk_uuid]}")

    # Load datasets and compute statistics
    df1 = task_dataframes[fn1]
    df2 = task_dataframes[fn2]
    df3 = task_dataframes[fn3]
    
    if range_to_analyze:
        start_date = pd.to_datetime(range_to_analyze[0])
        end_date = pd.to_datetime(range_to_analyze[1])
        
        mask1 = (df1['dt'] > start_date) & (df1['dt'] <= end_date)
        mask2 = (df2['dt'] > start_date) & (df2['dt'] <= end_date)
        mask3 = (df3['dt'] > start_date) & (df3['dt'] <= end_date)
        
        df1 = df1[mask1]
        df2 = df2[mask2]
        df3 = df3[mask3]
        
        df1 = df1.reset_index()
        df2 = df2.reset_index()
        df3 = df3.reset_index()
        
    res1, peaks1 = extract_activities(df1, se_radius, height_quantile)
    res2, peaks2 = extract_activities(df2, se_radius, height_quantile)
    res3, peaks3 = extract_activities(df3, se_radius, height_quantile)
    
    df_avg = pd.merge_asof(df1, df2, on='dt')
    df_avg = pd.merge_asof(df_avg, df3, on='dt')
    data_avg = {'data': df_avg[["data_x", "data_y", "data"]].mean(axis=1), 'dt': df_avg.dt}
    df_avg = pd.DataFrame(data=data_avg)

    df_med = pd.merge_asof(df1, df2, on='dt')
    df_med = pd.merge_asof(df_med, df3, on='dt')
    data_med = {'data': df_med[["data_x", "data_y", "data"]].median(axis=1), 'dt': df_med.dt}
    df_med = pd.DataFrame(data=data_med)
    
    df_max = pd.merge_asof(df1, df2, on='dt')
    df_max = pd.merge_asof(df_max, df3, on='dt')
    data_max = {'data': df_max[["data_x", "data_y", "data"]].max(axis=1), 'dt': df_max.dt}
    df_max = pd.DataFrame(data=data_max)
     
    #df_max['data3'] = df_avg.iloc[:, 1:4].max(axis=1)
    
    #df_avg = df_avg.drop(["data_x", "data_y", "data"], axis=1)
    #df_avg = df_avg.rename(columns={"data3": "data"})
    
    #df_avg = df_avg.drop(["data_x", "data_y", "data"], axis=1)
    #df_avg = df_avg.rename(columns={"data3": "data"})
    
    res_avg, peaks_avg = extract_activities(df_avg, se_radius, height_quantile)
    res_med, peaks_med = extract_activities(df_med, se_radius, height_quantile)
    res_max, peaks_max = extract_activities(df_max, se_radius, height_quantile)
    
    return {
        "df1": [df1, res1, peaks1],
        "df2": [df2, res2, peaks2],
        "df3": [df3, res3, peaks3],
        "avg": [df_avg, res_avg, peaks_avg],
        "med": [df_med, res_med, peaks_med],
        "max": [df_max, res_max, peaks_max],
    }

def analyze_walk(task_dataframes, participant_id, walk_uuid, trial_nr, se_radius=11):
    fn1 = f"{participant_id}_192.168.47.156_{walk_uuid}_{trial_nr}_flat.parquet"
    fn2 = f"{participant_id}_192.168.47.48_{walk_uuid}_{trial_nr}_flat.parquet"
    fn3 = f"{participant_id}_192.168.47.93_{walk_uuid}_{trial_nr}_flat.parquet"

    logging.info(f"{participant_id} - {TASKS[walk_uuid]}")

    # Load datasets and compute statistics
    df1 = task_dataframes[fn1]
    df2 = task_dataframes[fn2]
    df3 = task_dataframes[fn3]

    res1, peaks1 = extract_activities(df1, se_radius)
    res2, peaks2 = extract_activities(df2, se_radius)
    res3, peaks3 = extract_activities(df3, se_radius)

    logging.info(f"file {fn1}")
    compute_statistics(trail_df=df1, residuals=res1)

    logging.info(f"file {fn2}")
    compute_statistics(trail_df=df2, residuals=res2)

    logging.info(f"file {fn2}")
    compute_statistics(trail_df=df3, residuals=res3)

    logging.info("\n")
    
def get_range(participant_id, walkway_walk, trial_nr):
    try:
        return ranges_to_analyze[participant_id][f"{walkway_walk}-{trial_nr}"]
    except RuntimeError:
        return None
    
def get_experiment_values_for(participant_id, experiment_nr):
    experiment_data = list(ranges_to_analyze[participant_id].keys())[experiment_nr]
    return [int(k) for k in experiment_data.split("-")]
