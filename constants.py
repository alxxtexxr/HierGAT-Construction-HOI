ACTION_CLASSES = [
    # human
    'supervise',        # 0
    'collaborate with', # 1
    'assist',           # 2
    'lead',             # 3
    'coordinate with',  # 4
    'listen to',        # 5

    # rebar
    'prepare_rebar',    # 6 -> 'tie'
    'erect',            # 7
    'prepare_rebar',    # 8
    'carry',            # 9 -> 'transport'

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
    None,               # 17 -> 'no interaction'
]
NEW_ACTION_CLASSES = [
    # rebar
    'erect',            # 0
    'prepare_rebar',    # 1

    # equipment
    'use',              # 2
    'carry',            # 3

    # all
    'inspect',          # 4

    # renamed
    # 'tie',              # 5 -> 1
    # 'transport',        # 6 -> 3

    # removed
    # 'no interaction',   # 7
]
VIS_ACTION_CLASSES = [
    # rebar
    'erect',            # 0
    'prepare',          # 1

    # equipment
    'use',              # 2
    'carry',            # 3

    # all
    'inspect',          # 4
]
