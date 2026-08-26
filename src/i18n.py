"""
i18n.py
-------
Multilingual support.

Supported:
- English
- Hindi
- Kannada
- Tamil
- Telugu
- Malayalam

Handles:
- Language detection
- Translation
- Speech-to-text
- Text-to-speech
"""

import io

from deep_translator import (
    GoogleTranslator,
)

from gtts import (
    gTTS,
)

import speech_recognition as sr

from langdetect import (
    detect,
    DetectorFactory,
)

# DETERMINISTIC DETECTION

DetectorFactory.seed = 0


# SUPPORTED LANGUAGES

SUPPORTED_LANGUAGES = {

    "en":
        "English",

    "hi":
        "हिंदी (Hindi)",

    "kn":
        "ಕನ್ನಡ (Kannada)",

    "ta":
        "தமிழ் (Tamil)",

    "te":
        "తెలుగు (Telugu)",

    "ml":
        "മലയാളം (Malayalam)",
}


# SPEECH RECOGNITION LANGUAGES

_STT_CANDIDATES = [

    "hi-IN",

    "kn-IN",

    "ta-IN",

    "te-IN",

    "ml-IN",

    "en-IN",
]

# TEXT LANGUAGE DETECTION

def detect_language(
    text: str
) -> str:

    if not text or not text.strip():
        return "en"


    try:

        detected = detect(
            text
        )

    except Exception:

        return "en"


    if detected in SUPPORTED_LANGUAGES:

        return detected


    return "en"


# TRANSLATION

def translate_text(
    text: str,
    source: str,
    target: str,
) -> str:

    if (
        not text
        or not text.strip()
        or source == target
    ):

        return text


    try:

        translator = (
            GoogleTranslator(

                source=source,

                target=target,
            )
        )


        return translator.translate(
            text
        )


    except Exception:

        return text


# TEXT TO SPEECH

def text_to_speech(
    text: str,
    lang_code: str,
):

    if not text or not text.strip():

        return None


    try:

        tts = gTTS(

            text=text,

            lang=lang_code,
        )


        buffer = io.BytesIO()


        tts.write_to_fp(
            buffer
        )


        buffer.seek(0)


        return buffer.read()


    except Exception:

        return None


# SPEECH TO TEXT

def speech_to_text_auto(
    audio_bytes: bytes
):

    recognizer = (
        sr.Recognizer()
    )

    # READ AUDIO

    try:

        audio_file = (
            io.BytesIO(
                audio_bytes
            )
        )


        with sr.AudioFile(
            audio_file
        ) as source:

            audio_data = (
                recognizer.record(
                    source
                )
            )


    except Exception:

        return (
            None,
            None,
        )


    # TRY LANGUAGES

    first_successful = None


    for language_code in (
        _STT_CANDIDATES
    ):

        expected_language = (
            language_code.split(
                "-"
            )[0]
        )


        try:

            transcript = (
                recognizer.recognize_google(
                    audio_data,
                    language=language_code,
                )
            )


        except (
            sr.UnknownValueError,
            sr.RequestError,
        ):

            continue


        if not transcript:

            continue

        # FIRST SUCCESS

        if first_successful is None:

            first_successful = (
                transcript,
                expected_language,
            )


        # SHORT ENGLISH

        word_count = len(
            transcript.split()
        )


        if (
            expected_language == "en"
            and word_count <= 4
        ):

            return (
                transcript,
                "en",
            )


        # VERIFY DETECTED LANGUAGE

        try:

            detected = detect(
                transcript
            )

        except Exception:

            detected = (
                expected_language
            )


        if (
            detected
            == expected_language
        ):

            return (
                transcript,
                expected_language,
            )

    # FALLBACK

    if first_successful:

        return first_successful


    return (
        None,
        None,
    )