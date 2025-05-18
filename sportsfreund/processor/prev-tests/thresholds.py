
def get_thresholds_squats():

    _ANGLE_HIP_KNEE_VERT = {
                            'NORMAL' : (0,  15),
                            'TRANS'  : (35, 60),
                            'PASS'   : (65, 180)
                           }    

    thresholds = {
                    'HIP_KNEE_VERT': _ANGLE_HIP_KNEE_VERT,

                    'HIP_THRESH'   : [10, 50],
                    'ANKLE_THRESH' : 45,
                    'KNEE_THRESH'  : [50, 65, 180],

                    'OFFSET_THRESH'    : 50.0,
                    'INACTIVE_THRESH'  : 15.0,

                    'CNT_FRAME_THRESH' : 50
                            
                }

    return thresholds