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

NEW_ACTION_CLASSES_V2 = [
    # rebar
    'erect',            # 0
    'prepare_rebar',    # 1

    # equipment         
    'use',              # 2
    'carry',            # 3

    # all
    'inspect',          # 4
    
    # replaced
    'tie',              # 5 -> 1
    'transport',        # 6 -> 3
    
    # removed
    'no interaction',   # 7
]
VIS_ACTION_CLASSES_V2 = [
    # rebar
    'erect',            # 0
    'prepare',          # 1

    # equipment         
    'use',              # 2
    'carry',            # 3

    # all
    'inspect',          # 4
]

FEATURE_DIRS = [
    '/root/vs-gats-plaster/deepsort/hiergat_data_v3_3s/C0071_full_MP4_anno_for_labelling_done_faridz_full_temporal_3s/features',
    '/root/vs-gats-plaster/deepsort/hiergat_data_v3_3s/C0074_full_MP4_anno_for_labelling_done_ray_full_temporal_3s/features',   
    '/root/vs-gats-plaster/deepsort/hiergat_data_v3_3s/C0078_full_MP4_anno_for_labelling_done_anne_full_temporal_3s/features',  
    '/root/vs-gats-plaster/deepsort/hiergat_data_v3_3s/C0085_full_MP4_anno_for_labelling_done_yoga_full_temporal_3s/features',  
    '/root/vs-gats-plaster/deepsort/hiergat_data_v3_3s/C0087_full_MP4_anno_for_labelling_done_arga_full_temporal_3s/features',  
    '/root/vs-gats-plaster/deepsort/hiergat_data_v3_3s/C0090_full_MP4_anno_for_labelling_rizky_full_temporal_3s/features',       
    '/root/vs-gats-plaster/deepsort/hiergat_data_v3_3s/C0098_full_MP4_anno_for_labelling_done_akbar_full_temporal_3s/features', 
    '/root/vs-gats-plaster/deepsort/hiergat_data_v3_3s/C0101_full_MP4_anno_for_labelling_done_faridz_full_temporal_3s/features',
    '/root/vs-gats-plaster/deepsort/hiergat_data_v3_3s/C0105_full_MP4_anno_for_labelling_full_temporal_3s/features',             
    '/root/vs-gats-plaster/deepsort/hiergat_data_v3_3s/C0106_full_MP4_anno_for_labelling_full_temporal_3s/features',             
    '/root/vs-gats-plaster/deepsort/hiergat_data_v3_3s/C0108_full_MP4_anno_for_labelling_fixed_full_temporal_3s/features',      
    '/root/vs-gats-plaster/deepsort/hiergat_data_v3_3s/C0110_full_MP4_anno_for_labelling_full_temporal_3s/features',            
    '/root/vs-gats-plaster/deepsort/hiergat_data_v3_3s/C0100_full_MP4_anno_for_labelling_done_putu_full_temporal_3s/features',  
    '/root/vs-gats-plaster/deepsort/hiergat_data_v3_3s/C0109_full_MP4_anno_for_labelling_full_temporal_3s/features',            
    '/root/vs-gats-plaster/deepsort/hiergat_data_v3_3s/C0104_full_MP4_anno_for_labelling_full_temporal_3s/features',             
    '/root/vs-gats-plaster/deepsort/hiergat_data_v3_3s/C0099_full_MP4_anno_for_labelling_done_arga_full_temporal_3s/features',  
    '/root/vs-gats-plaster/deepsort/hiergat_data_v3_3s/C0083_full_MP4_anno_for_labelling_done_putu_full_temporal_3s/features',  
]
FEATURE_DIRS = [Path(p) for p in FEATURE_DIRS]