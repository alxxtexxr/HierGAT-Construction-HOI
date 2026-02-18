from pathlib import Path

ACTION_CLASSES = [
    # human
    'supervise',        # 0
    'collaborate with', # 1
    'assist',           # 2
    'lead',             # 3
    'coordinate with',  # 4
    'listen to',        # 5

    # rebar
    'tie',              # 6
    'erect',            # 7
    'prepare_rebar',    # 8
    'transport',        # 9

    # formwork
    'install',          # 10
    'prepare_formwork', # 11

    # concrete
    'pour',             # 12
    'finish',           # 13

    # equipment         
    'use',              # 14
    'carry',            # 15

    # all
    'inspect',          # 16
    'no interaction',   # 17
]
NEW_ACTION_CLASSES = [
    # rebar
    'tie',              # 0
    'erect',            # 1
    'prepare_rebar',    # 2
    'transport',        # 3

    # equipment         
    'use',              # 4
    'carry',            # 5

    # all
    'inspect',          # 6
    'no interaction',   # 7
]
VIS_ACTION_CLASSES = [
    # rebar
    'tie',              # 0
    'erect',            # 1
    'prepare',          # 2
    'transport',        # 3

    # equipment         
    'use',              # 4
    'carry',            # 5

    # all
    'inspect',          # 6
    'no interaction',   # 7
]

FEATURE_DIRS = [
    '/root/vs-gats-plaster/deepsort/hiergat_data_v3_3s/C0071_full_MP4_anno_for_labelling_done_faridz_full_temporal_3s/features', # {0: 130, 2: 90, 6: 105, 7: 223}
    '/root/vs-gats-plaster/deepsort/hiergat_data_v3_3s/C0074_full_MP4_anno_for_labelling_done_ray_full_temporal_3s/features',    # {0: 124, 1: 25, 2: 11, 6: 55, 7: 133}
    '/root/vs-gats-plaster/deepsort/hiergat_data_v3_3s/C0078_full_MP4_anno_for_labelling_done_anne_full_temporal_3s/features',   # {0: 130, 7: 65} 
    '/root/vs-gats-plaster/deepsort/hiergat_data_v3_3s/C0085_full_MP4_anno_for_labelling_done_yoga_full_temporal_3s/features',   # {0: 200, 2: 181, 4: 27, 7: 84}
    '/root/vs-gats-plaster/deepsort/hiergat_data_v3_3s/C0087_full_MP4_anno_for_labelling_done_arga_full_temporal_3s/features',   # {0: 205, 5: 8, 6: 134, 7: 236}
    '/root/vs-gats-plaster/deepsort/hiergat_data_v3_3s/C0090_full_MP4_anno_for_labelling_rizky_full_temporal_3s/features',       
    '/root/vs-gats-plaster/deepsort/hiergat_data_v3_3s/C0098_full_MP4_anno_for_labelling_done_akbar_full_temporal_3s/features',  # {0: 2, 1: 10, 5: 23, 7: 84}
    '/root/vs-gats-plaster/deepsort/hiergat_data_v3_3s/C0101_full_MP4_anno_for_labelling_done_faridz_full_temporal_3s/features', # {0: 147, 6: 5, 7: 309}
    '/root/vs-gats-plaster/deepsort/hiergat_data_v3_3s/C0105_full_MP4_anno_for_labelling_full_temporal_3s/features',             
    '/root/vs-gats-plaster/deepsort/hiergat_data_v3_3s/C0106_full_MP4_anno_for_labelling_full_temporal_3s/features',             
    '/root/vs-gats-plaster/deepsort/hiergat_data_v3_3s/C0108_full_MP4_anno_for_labelling_fixed_full_temporal_3s/features',       # {1: 26, 3: 21, 4: 35, 6: 16, 7: 157}
    '/root/vs-gats-plaster/deepsort/hiergat_data_v3_3s/C0110_full_MP4_anno_for_labelling_full_temporal_3s/features',             # {2: 23, 3: 29, 4: 25, 7: 264}
    '/root/vs-gats-plaster/deepsort/hiergat_data_v3_3s/C0100_full_MP4_anno_for_labelling_done_putu_full_temporal_3s/features',   # {0: 23, 2: 10, 4: 2, 7: 191}
    '/root/vs-gats-plaster/deepsort/hiergat_data_v3_3s/C0109_full_MP4_anno_for_labelling_full_temporal_3s/features',             # {1: 1541, 3: 38, 4: 35}       
    '/root/vs-gats-plaster/deepsort/hiergat_data_v3_3s/C0104_full_MP4_anno_for_labelling_full_temporal_3s/features',             
    '/root/vs-gats-plaster/deepsort/hiergat_data_v3_3s/C0099_full_MP4_anno_for_labelling_done_arga_full_temporal_3s/features',   # {0: 4, 2: 274, 5: 5, 6: 6, 7: 25}
    '/root/vs-gats-plaster/deepsort/hiergat_data_v3_3s/C0083_full_MP4_anno_for_labelling_done_putu_full_temporal_3s/features',   # {0: 240, 1: 5, 2: 56, 3: 1, 4: 6, 6: 2, 7: 34}
]
FEATURE_DIRS = [Path(p) for p in FEATURE_DIRS]