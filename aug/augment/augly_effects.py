import augly.audio as A

ARTEFACT_EFFECTS = [
    A.Compose([
            A.ChangeVolume(
                volume_db = 8,
                p = 1
            ),
            A.PeakingEqualizer (
                center_hz = 120,
                q = 0.7,
                gain_db = 12,
                p = 1
            ),
            A.PeakingEqualizer (
                center_hz = 150,
                q = 1.0,
                gain_db = 6,
                p = 1
            ),
        ]),

        A.Compose([
            A.Reverb(
                reverberance = 60,
                room_scale = 40.0,
                p = 1
            ),
            A.HighPassFilter(
                cutoff_hz = 150,
                p = 1
            ),
            
        ]),

        A.Compose([    
            A.LowPassFilter(
                cutoff_hz=500,
                p=1.0
            ),
        ]),

        A.Compose([
            A.PeakingEqualizer (
                center_hz = 8000,
                q = 2,
                gain_db = -6,
                p = 1
            ),
        ]),

        A.Compose([
            A.Harmonic(
                kernel_size = 31,
                power = 2,
                margin = 1.5,
                p = 1
            ),
        ],p = 1),

        A.Compose([
            A.PitchShift(
                n_steps = 5,
                p = 1
            ),
        ]),

        A.Compose([
            A.Percussive(
                kernel_size = 31,
                power = 2,
                margin = 1,
                p = 1
            ),
        ]),

        A.Compose([
                A.TimeStretch(
                rate = 1.5,
                p = 1
            ),
        ]),

        A.Compose([
            A.Tempo(
                factor = 1.5,
                p = 1
            ),
        ])
]