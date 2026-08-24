"""Pinned model revisions shared by setup and release verification."""

COMMON_MODELS = (
    ("Qwen/Qwen3-TTS-12Hz-1.7B-Base", "fd4b254389122332181a7c3db7f27e918eec64e3", "qwen3-tts/models/Qwen3-TTS-12Hz-1.7B-Base"),
    ("Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign", "5ecdb67327fd37bb2e042aab12ff7391903235d3", "qwen3-tts/models/Qwen3-TTS-12Hz-1.7B-VoiceDesign"),
    ("Systran/faster-whisper-large-v3", "edaa852ec7e145841d8ffdb056a99866b5f0a478", "asr/models/faster-whisper-large-v3"),
)

TRANSLATION_MODELS = {
    "cuda": ("tencent/Hy-MT2-7B-FP8", "883d09eb21d9be92058556cd0a4016d8a648c7db", "hymt/models/Hy-MT2-7B-FP8"),
    "7b": ("tencent/Hy-MT2-7B", "9b0eb4e8f001def3e5ff6469a0ac96fdb39ec223", "hymt/models/Hy-MT2-7B"),
    "1.8b": ("tencent/Hy-MT2-1.8B", "9a341cd1b679d3efd23b46e847b01745a71ed792", "hymt/models/Hy-MT2-1.8B"),
}

SEED_MODELS = (
    ("Plachta/Seed-VC", "257283f9f41585055e8f858fba4fd044e5caed6e"),
    ("funasr/campplus", "e4b6ede7ce16997aff4ae69fbca1f0175e2afede"),
    ("nvidia/bigvgan_v2_22khz_80band_256x", "633ff708ed5b74903e86ff1298cf4a98e921c513"),
    ("openai/whisper-small", "973afd24965f72e36ca33b3055d56a652f456b4d"),
)

MUSETALK_MODELS = (
    ("TMElyralab/MuseTalk", "3ef28bc5cff08c90ad8178a25f1b570cd800170f"),
    ("stabilityai/sd-vae-ft-mse", "31f26fdeee1355a5c34592e401dd41e45d25a493"),
    ("openai/whisper-tiny", "169d4a4341b33bc18d8881c4b69c2e104e1cc0af"),
    ("yzd-v/DWPose", "1a7144101628d69ee7a3768d1ee3a094070dc388"),
    ("ManyOtherFunctions/face-parse-bisent", "0073b233a5a3c4b1377d4dbf49245017938a72b5"),
)
